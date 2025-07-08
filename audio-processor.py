import json
import boto3
import os
import time
import uuid
import re
import urllib.parse
import logging
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Configuration constants
class Config:
    """Configuration settings for the audio processing pipeline"""
    REGION_NAME = "us-west-2"
    PROJECT_NAME_PREFIX = 'bda-audio-project'
    MAX_POLL_ATTEMPTS = 60
    POLL_INTERVAL = 5  # seconds
    MAX_TRANSCRIPT_LENGTH = 100000
    MAX_BEDROCK_RETRIES = 3
    SUPPORTED_FILE_EXTENSIONS = ['.mp3']

class AudioProcessor:
    """Main class for processing audio files through AWS Bedrock Data Automation"""
    
    def __init__(self):
        """Initialize AWS clients and configuration"""
        self.s3_client = None
        self.bda_client = None
        self.bda_runtime_client = None
        self.bedrock_runtime = None
        self.account_id = None
        self._initialize_aws_clients()
    
    def _initialize_aws_clients(self) -> None:
        """Initialize AWS clients with proper error handling"""
        try:
            logger.info("Initializing AWS clients")
            
            self.s3_client = boto3.client('s3')
            self.bda_client = boto3.client('bedrock-data-automation', region_name=Config.REGION_NAME)
            self.bda_runtime_client = boto3.client('bedrock-data-automation-runtime', region_name=Config.REGION_NAME)
            self.bedrock_runtime = boto3.client('bedrock-runtime', region_name=Config.REGION_NAME)
            
            # Get account ID for ARN construction
            sts = boto3.client('sts')
            self.account_id = sts.get_caller_identity()["Account"]
            
            logger.info(f"AWS clients initialized successfully for account: {self.account_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize AWS clients: {str(e)}")
            raise
    
    def process_audio_file(self, event: Dict, context: Any) -> Dict:
        """
        Main processing method for audio files
        
        Args:
            event: S3 event trigger
            context: Lambda context
            
        Returns:
            Dict: Processing results and status
        """
        start_time = time.time()
        self._log_execution_context(context)
        
        try:
            # Extract S3 event information
            file_info = self._extract_s3_event_info(event)
            logger.info(f"Processing file: {file_info['file_name']} ({file_info['file_size']} bytes)")
            
            # Skip non-supported files
            if not self._is_supported_file(file_info['file_name']):
                return self._create_response(200, f"Skipped unsupported file: {file_info['file_name']}")
            
            # Process through BDA pipeline
            processing_results = self._run_bda_pipeline(file_info, context)
            
            # Generate metadata using Bedrock
            metadata = self._generate_metadata(file_info['file_name'], processing_results['transcript'])
            
            # Save results to S3
            output_files = self._save_results_to_s3(
                file_info, 
                processing_results['bda_result'], 
                metadata
            )
            
            processing_time = time.time() - start_time
            logger.info(f"Processing completed successfully in {processing_time:.2f} seconds")
            
            return self._create_response(200, {
                'message': 'Audio processing completed successfully',
                'files': output_files,
                'processing_time_seconds': round(processing_time, 2)
            })
            
        except Exception as e:
            logger.error(f"Error processing audio file: {str(e)}")
            return self._create_response(500, f'Error processing audio file: {str(e)}')
    
    def _extract_s3_event_info(self, event: Dict) -> Dict:
        """Extract file information from S3 event"""
        if 'Records' not in event or len(event['Records']) == 0:
            raise ValueError("No valid S3 event records found")
        
        record = event['Records'][0]['s3']
        
        return {
            'bucket_name': record['bucket']['name'],
            'object_key': urllib.parse.unquote_plus(record['object']['key']),
            'file_name': os.path.basename(urllib.parse.unquote_plus(record['object']['key'])),
            'file_size': record['object']['size']
        }
    
    def _is_supported_file(self, filename: str) -> bool:
        """Check if file extension is supported"""
        file_ext = os.path.splitext(filename)[1].lower()
        return file_ext in Config.SUPPORTED_FILE_EXTENSIONS
    
    def _run_bda_pipeline(self, file_info: Dict, context: Any) -> Dict:
        """Run the complete BDA processing pipeline"""
        # Get or create BDA project
        project_arn = self._get_or_create_bda_project()
        
        # Submit file for processing
        invocation_arn, output_path = self._submit_file_to_bda(file_info, project_arn)
        
        # Wait for completion with dynamic timeout
        max_attempts = self._calculate_max_poll_attempts(context)
        self._wait_for_bda_completion(invocation_arn, max_attempts)
        
        # Retrieve results
        transcript, bda_result = self._get_bda_results(file_info['bucket_name'], output_path)
        
        return {
            'transcript': transcript,
            'bda_result': bda_result,
            'output_path': output_path
        }
    
    def _get_or_create_bda_project(self) -> str:
        """Get existing BDA project or create new one"""
        try:
            # Look for existing projects
            projects_response = self.bda_client.list_data_automation_projects()
            existing_projects = [
                p for p in projects_response.get('projects', [])
                if p["projectName"].startswith(Config.PROJECT_NAME_PREFIX)
            ]
            
            if existing_projects:
                project_arn = existing_projects[0]["projectArn"]
                logger.info(f"Using existing BDA project: {project_arn}")
                return project_arn
            
            # Create new project
            return self._create_new_bda_project()
            
        except Exception as e:
            logger.error(f"Error managing BDA project: {str(e)}")
            raise
    
    def _create_new_bda_project(self) -> str:
        """Create a new BDA project with audio processing configuration"""
        project_name = f"{Config.PROJECT_NAME_PREFIX}-{str(uuid.uuid4())[0:8]}"
        
        project_config = {
            'projectName': project_name,
            'projectDescription': 'Audio processing project for automated transcription and analysis',
            'projectStage': 'DEVELOPMENT',
            'standardOutputConfiguration': {
                "audio": {
                    "extraction": {
                        "category": {
                            "state": "ENABLED", 
                            "types": ["AUDIO_CONTENT_MODERATION", "TOPIC_CONTENT_MODERATION", "TRANSCRIPT"]
                        }
                    },
                    "generativeField": {
                        "state": "ENABLED",
                        "types": ["AUDIO_SUMMARY", "TOPIC_SUMMARY", "IAB"]
                    }
                }
            }
        }
        
        response = self.bda_client.create_data_automation_project(**project_config)
        project_arn = response.get("projectArn")
        
        logger.info(f"Created new BDA project: {project_name} (ARN: {project_arn})")
        return project_arn
    
    def _submit_file_to_bda(self, file_info: Dict, project_arn: str) -> Tuple[str, str]:
        """Submit audio file to BDA for processing"""
        # Verify input file exists
        self._verify_input_file(file_info['bucket_name'], file_info['object_key'])
        
        # Create unique output path
        base_name = os.path.splitext(file_info['file_name'])[0]
        output_path = f'outputs/bda_processing/{base_name}/{str(uuid.uuid4())[0:8]}/'
        
        # Submit to BDA
        profile_arn = f'arn:aws:bedrock:{Config.REGION_NAME}:{self.account_id}:data-automation-profile/us.data-automation-v1'
        
        submission_config = {
            'inputConfiguration': {'s3Uri': f"s3://{file_info['bucket_name']}/{file_info['object_key']}"},
            'outputConfiguration': {'s3Uri': f"s3://{file_info['bucket_name']}/{output_path}"},
            'dataAutomationProfileArn': profile_arn,
            'dataAutomationConfiguration': {
                'dataAutomationProjectArn': project_arn,  
                'stage': 'DEVELOPMENT'
            }
        }
        
        response = self.bda_runtime_client.invoke_data_automation_async(**submission_config)
        invocation_arn = response.get("invocationArn")
        
        if not invocation_arn:
            raise Exception("Failed to get invocationArn from BDA API response")
        
        logger.info(f"BDA processing started - Invocation ARN: {invocation_arn}")
        return invocation_arn, output_path
    
    def _wait_for_bda_completion(self, invocation_arn: str, max_attempts: int) -> None:
        """Poll BDA service until processing completes"""
        attempts = 0
        
        while attempts < max_attempts:
            try:
                status_response = self.bda_runtime_client.get_data_automation_status(
                    invocationArn=invocation_arn
                )
                status = status_response.get("status")
                
                logger.info(f"BDA status check {attempts + 1}/{max_attempts}: {status}")
                
                if status == "Success":
                    logger.info("BDA processing completed successfully")
                    return
                elif status in ["ServiceError", "ClientError"]:
                    error_details = status_response.get("errorDetails", "No error details provided")
                    raise Exception(f"BDA processing failed with status {status}: {error_details}")
                
                time.sleep(Config.POLL_INTERVAL)
                attempts += 1
                
            except Exception as e:
                if "ServiceError" in str(e) or "ClientError" in str(e):
                    raise  # Re-raise BDA errors immediately
                logger.warning(f"Error checking BDA status: {str(e)}")
                attempts += 1
                time.sleep(Config.POLL_INTERVAL)
        
        raise Exception(f"BDA processing timed out after {max_attempts} attempts")
    
    def _get_bda_results(self, bucket_name: str, output_path: str) -> Tuple[str, Dict]:
        """Retrieve and parse BDA processing results"""
        # Find result file
        all_objects = self._list_s3_objects(bucket_name, output_path)
        result_file = next((obj['Key'] for obj in all_objects if 'result.json' in obj['Key']), None)
        
        if not result_file:
            raise Exception(f"No result.json found in {output_path}")
        
        # Download and parse results
        result_obj = self.s3_client.get_object(Bucket=bucket_name, Key=result_file)
        result_data = json.loads(result_obj['Body'].read().decode('utf-8'))
        
        # Extract transcript
        transcript = self._extract_transcript_from_result(result_data)
        
        logger.info(f"Retrieved BDA results - Transcript length: {len(transcript)} characters")
        return transcript, result_data
    
    def _extract_transcript_from_result(self, result_data: Dict) -> str:
        """Extract transcript text from BDA result data"""
        try:
            return result_data['audio']['transcript']['representation'].get('text', '')
        except KeyError:
            logger.warning("No transcript found in BDA result")
            return ""
    
    def _generate_metadata(self, filename: str, transcript: str) -> Dict:
        """Generate metadata combining filename parsing and Bedrock analysis"""
        # Parse filename for structured metadata
        filename_metadata = FilenameParser.parse_audio_filename(filename)
        
        # Analyze transcript with Bedrock
        bedrock_analysis = TranscriptAnalyzer(self.bedrock_runtime).analyze_transcript(transcript)
        
        return {
            **filename_metadata,
            "bedrock_analysis": bedrock_analysis,
            "processing_timestamp": datetime.utcnow().isoformat() + "Z",
            "transcript_length": len(transcript)
        }
    
    def _save_results_to_s3(self, file_info: Dict, bda_result: Dict, metadata: Dict) -> Dict:
        """Save processing results to S3 output folder"""
        base_name = os.path.splitext(file_info['file_name'])[0]
        
        output_files = {
            'bda_result': f"output/{base_name}_result.json",
            'metadata': f"output/{base_name}_metadata.json"
        }
        
        # Save BDA result
        self.s3_client.put_object(
            Bucket=file_info['bucket_name'],
            Key=output_files['bda_result'],
            Body=json.dumps(bda_result, indent=2),
            ContentType='application/json'
        )
        
        # Save metadata
        self.s3_client.put_object(
            Bucket=file_info['bucket_name'],
            Key=output_files['metadata'],
            Body=json.dumps(metadata, indent=2),
            ContentType='application/json'
        )
        
        logger.info(f"Results saved to S3: {output_files}")
        return output_files
    
    # Utility methods
    def _verify_input_file(self, bucket: str, key: str) -> None:
        """Verify that input file exists and is accessible"""
        try:
            self.s3_client.head_object(Bucket=bucket, Key=key)
        except Exception as e:
            raise Exception(f"Input file s3://{bucket}/{key} does not exist or is not accessible: {str(e)}")
    
    def _list_s3_objects(self, bucket: str, prefix: str) -> list:
        """List all objects with pagination support"""
        result = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            if 'Contents' in page:
                result.extend(page['Contents'])
        return result
    
    def _calculate_max_poll_attempts(self, context: Any) -> int:
        """Calculate maximum polling attempts based on remaining execution time"""
        remaining_time_seconds = context.get_remaining_time_in_millis() / 1000
        buffer_time = 30  # Reserve 30 seconds for cleanup
        return max(5, int((remaining_time_seconds - buffer_time) / Config.POLL_INTERVAL))
    
    def _log_execution_context(self, context: Any) -> None:
        """Log Lambda execution context for debugging"""
        logger.info(f"Lambda execution started - Request ID: {context.aws_request_id}")
        logger.info(f"Memory limit: {context.memory_limit_in_mb}MB")
        logger.info(f"Remaining time: {context.get_remaining_time_in_millis()}ms")
    
    def _create_response(self, status_code: int, body: Any) -> Dict:
        """Create standardized Lambda response"""
        return {
            'statusCode': status_code,
            'body': json.dumps(body) if isinstance(body, dict) else body
        }


class FilenameParser:
    """Utility class for parsing audio filenames"""
    
    @staticmethod
    def parse_audio_filename(filename: str) -> Dict:
        """
        Parse audio filename following pattern: MMM_CCCC_SS_AWS-RR-YYYY-MM-DD_HH-MM-SS-mmm.mp3
        
        Returns:
            Dict: Parsed metadata or error information
        """
        logger.info(f"Parsing filename: {filename}")
        
        pattern = r"([A-Z]{3})_([A-Z]{4})_([A-Z]{2})_AWS-(\d{2})-(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})-(\d{3})\.mp3"
        match = re.match(pattern, filename)
        
        if not match:
            return {
                "original_filename": filename,
                "parsing_status": "failed",
                "parsing_error": "Filename does not match expected pattern"
            }
        
        groups = match.groups()
        return {
            "market_abbreviation": groups[0],
            "callsign": groups[1],
            "service_type": groups[2],
            "recording_source": "AWS",
            "recording_deck": groups[3],
            "date": f"{groups[4]}-{groups[5]}-{groups[6]}",
            "time": f"{groups[7]}:{groups[8]}:{groups[9]}.{groups[10]}",
            "timestamp_components": {
                "year": groups[4],
                "month": groups[5],
                "day": groups[6],
                "hour": groups[7],
                "minute": groups[8],
                "second": groups[9],
                "millisecond": groups[10]
            },
            "original_filename": filename,
            "parsing_status": "success"
        }


class TranscriptAnalyzer:
    """Handles transcript analysis using Amazon Bedrock"""
    
    def __init__(self, bedrock_runtime):
        self.bedrock_runtime = bedrock_runtime
    
    def analyze_transcript(self, transcript: str) -> Dict:
        """Analyze transcript using Claude model via Bedrock"""
        if not transcript:
            return {"analysis_status": "failed", "reason": "No transcript available"}
        
        # Truncate if necessary
        if len(transcript) > Config.MAX_TRANSCRIPT_LENGTH:
            transcript = transcript[:Config.MAX_TRANSCRIPT_LENGTH] + "... [TRUNCATED]"
            logger.warning(f"Transcript truncated to {Config.MAX_TRANSCRIPT_LENGTH} characters")
        
        prompt = self._build_analysis_prompt(transcript)
        
        for attempt in range(Config.MAX_BEDROCK_RETRIES):
            try:
                return self._call_bedrock_model(prompt)
            except Exception as e:
                if "ThrottlingException" in str(e) and attempt < Config.MAX_BEDROCK_RETRIES - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Bedrock throttling, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Bedrock analysis failed: {str(e)}")
                    return {"analysis_status": "failed", "reason": str(e)}
        
        return {"analysis_status": "failed", "reason": "Max retries exceeded"}
    
    def _build_analysis_prompt(self, transcript: str) -> str:
        """Build the analysis prompt for Bedrock"""
        return f"""
        Analyze this audio transcript and provide answers in JSON format.
        
        Transcript:
        {transcript}
        
        Please analyze and return a JSON object with these fields:
        - talking_vs_other_ratio: Estimated percentage of talking vs music/commercials
        - segment_topics: Array of main topics discussed
        - speakers: Estimated number and types of speakers
        - segment_types: Classification of content (news, sports, entertainment, etc.)
        - commercial_count: Number of commercials detected
        - advertisers: List of advertisers mentioned
        - commercial_categories: Types of products/services advertised
        
        Respond only with valid JSON.
        """
    
    def _call_bedrock_model(self, prompt: str) -> Dict:
        """Make API call to Bedrock model"""
        request_body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 4000,
            "temperature": 0.5,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        }
        
        response = self.bedrock_runtime.invoke_model(
            modelId="anthropic.claude-3-haiku-20240307-v1:0",
            body=json.dumps(request_body)
        )
        
        model_response = json.loads(response["body"].read())
        response_text = model_response["content"][0]["text"]
        
        return self._extract_json_from_response(response_text)
    
    def _extract_json_from_response(self, response_text: str) -> Dict:
        """Extract JSON from model response"""
        # Try code blocks first
        code_blocks = re.findall(r'```(?:json)?(.*?)```', response_text, re.DOTALL)
        if code_blocks:
            return json.loads(code_blocks[0].strip())
        
        # Try finding JSON object
        json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # If no JSON found, return error with sample
        return {
            "analysis_status": "failed",
            "reason": "Could not extract JSON from response",
            "raw_response": response_text[:500] + "..." if len(response_text) > 500 else response_text
        }


# Lambda entry point
def lambda_handler(event, context):
    """
    AWS Lambda entry point for audio processing
    
    Processes audio files uploaded to S3 through:
    1. AWS Bedrock Data Automation for transcription and analysis
    2. Amazon Bedrock for intelligent content analysis
    3. Structured metadata extraction from filenames
    """
    processor = AudioProcessor()
    return processor.process_audio_file(event, context)