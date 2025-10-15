# AI-Powered Personality Insights Platform

## Project Overview
An advanced AI-powered personality insights platform that combines cutting-edge technologies for comprehensive emotional and visual analysis with enhanced cognitive profiling capabilities.

### Key Features
- Multi-modal analysis (images, videos, documents, text)
- Enhanced cognitive profiling with intelligence assessment
- Evidence-based psychological analysis with direct quotations
- Multi-service fallback chains for reliability
- Multiple download formats (PDF, Word, TXT)
- DeepSeek as default AI model with expanded LLM options
- Real-time chat interface for follow-up questions

### Technology Stack
- **Frontend**: React.js with TypeScript, Tailwind CSS, shadcn/ui
- **Backend**: Express.js with TypeScript
- **AI Models**: DeepSeek (default), OpenAI GPT-4o, Anthropic Claude, Perplexity
- **Analysis Services**: 
  - Facial Analysis: Azure Face API, Face++, AWS Rekognition
  - Video Analysis: Azure Video Indexer
  - Audio Transcription: Gladia, AssemblyAI, Deepgram, OpenAI Whisper
- **Storage**: In-memory storage with structured data models

## Recent Changes

### Fixed 50-Question Framework Implementation (October 15, 2025)
✓ COMPLETELY REPLACED DEFUNCT 20/40/60 QUESTION SYSTEM WITH DEDICATED 50-QUESTION FRAMEWORKS
✓ Implemented three specialized 50-question frameworks, one for each analysis type:
  - **Photo Analysis**: 50 questions covering Physical Cues, Expression & Emotion, Composition & Context, Personality & Psychological Inference, Symbolic & Metapsychological Analysis
  - **Video Analysis**: 50 questions covering Physical & Behavioral Cues, Expression & Emotion Over Time, Speech/Voice/Timing, Context/Environment/Interaction, Personality & Psychological Inference
  - **Text Analysis**: 50 questions covering Language & Style, Emotional Indicators, Cognitive & Structural Patterns, Self-Representation & Identity, Symbolic & Unconscious Material
✓ Removed analysisDepth parameter from all backend routes, schemas, and frontend components
✓ Removed Analysis Depth Selector UI component, updated step numbering (Step 3 → Step 2)
✓ Updated validation system to check for substantive content rather than specific field names
✓ All analyses now use fixed 50-question framework - no depth selection required
✓ Prompts dynamically select photo vs video questions based on media type
✓ Build successful with no errors, system fully operational with new framework

### Visual Description System Implementation - Complete Success (August 20, 2025)
✓ ELIMINATED PLACEHOLDER TEXT ISSUE by requiring detailed visual descriptions first
✓ Added mandatory visual description requirements to all image/video analysis prompts
✓ System now demands specific details: gender, age, body type, clothing, posture, facial expressions
✓ Required description of background, objects, hand positions, and specific actions observed
✓ All psychological assessments must be supported by the concrete visual details described
✓ Updated both single-person and multi-person analysis prompt structures
✓ Enhanced summary sections to start with visual description before psychological analysis
✓ Verified system working through successful image and video analysis testing
✓ Analyses now provide authentic visual observations instead of generic placeholder content

### Critical "Not Assessed" Fix - Complete Success (August 17, 2025)
✓ SUCCESSFULLY ELIMINATED ALL "Not assessed" RESPONSES from psychological analyses
✓ Updated image/video analysis prompts to explicitly forbid incomplete responses
✓ Enhanced AI prompts with aggressive requirements for substantive content on every question
✓ Forced AI models to make reasonable psychological inferences even with limited visual data
✓ Added critical requirement language: "Not assessed is NOT acceptable - provide answers to ALL questions"
✓ Verified system working perfectly through comprehensive testing with video and image analysis
✓ All psychological assessments now provide complete, substantive answers to every question

### Critical Text Analysis Fixes & API Key Reactivation (August 17, 2025)
✓ SUCCESSFULLY REACTIVATED ALL MAJOR API SERVICES with comprehensive key management
✓ Fixed critical text analysis parsing issues preventing proper JSON processing
✓ Enhanced text analysis prompt to generate 8-12 direct quotes and comprehensive 3-4 paragraph sections
✓ Added cleanMarkdownFromAnalysis function to remove all markdown formatting from AI responses
✓ Updated response formatting to display complete sections: Content Themes, Cognitive Profile, Speech Analysis, Values, etc.
✓ Fixed Anthropic API JSON parsing to handle code block wrapping and extract clean JSON
✓ Applied markdown cleaning across all AI models (OpenAI, Anthropic, Perplexity)
✓ Enhanced analysis depth requirements with specific evidence extraction and comprehensive psychological assessment

### API Services Now Active:
✓ 4 AI Models: Anthropic Claude (ZHI 1), OpenAI GPT-4o (ZHI 2), DeepSeek (ZHI 3), Perplexity (ZHI 4)
✓ 3 Transcription Services: Gladia, AssemblyAI, Deepgram
✓ 3 Facial Analysis Services: Face++, Azure Face, Google Vision
✓ Multi-service fallback chains operational
✓ Session management with SESSION_SECRET configured

### Anthropic Default Model & Comprehensive Analysis Enhancement (August 16, 2025)
✓ Changed default AI model from DeepSeek to Anthropic Claude across all schemas and frontend
✓ Fixed critical speech analysis issue where video transcriptions were being ignored
✓ Enhanced single-person analysis prompt to match multi-person speech-first analysis system
✓ Updated analysis to prioritize actual speech content over visual-only placeholder text
✓ Removed all markdown formatting (# and ###) from analysis output for cleaner presentation
✓ Updated all dropdown labels and service status displays to use ZHI naming convention
  - Anthropic → ZHI 1 (Default)
  - OpenAI → ZHI 2
  - DeepSeek → ZHI 3
  - Perplexity → ZHI 4
✓ ENHANCED ANALYSIS DEPTH: Made all first-time analyses comprehensive and deep by default
  - Increased quote extraction from 5-8 to 8-12 meaningful quotes
  - Enhanced analysis requirements for 3-4 paragraph sections with rich detail
  - Added comprehensive psychological depth requirements
  - Eliminated need for users to ask for "deeper" analysis - now default behavior
✓ Fixed chat system formatting consistency to match main analysis styling
✓ Removed debug text from user interface for cleaner presentation

### Video Segment Selection System Implementation (July 7, 2025)
✓ Successfully implemented complete video chunking system with user-selectable 3-second segments
✓ Added automatic video duration detection and segment validation
✓ Enhanced UI with real-time segment preview and processing time expectations
✓ Fixed PayloadTooLargeError with 50MB file size limits and early validation
✓ Added 10-minute server timeout for complex video processing
✓ Tested and verified: 6.74s video properly processed with 0-3s segment selection
✓ Confirmed multi-service fallback chains working (Face++, AssemblyAI transcription)

### Enhanced Speech/Text Content Integration (July 7, 2025)
✓ Dramatically enhanced analysis prompts to prioritize speech and text content
✓ Added comprehensive content analysis that discusses what people actually say/write
✓ Enhanced cognitive profiling based on vocabulary, reasoning patterns, and idea sophistication
✓ Increased direct quotations from 3-5 to 5-8 meaningful examples
✓ Added content themes analysis revealing interests, expertise, and priorities
✓ Enhanced personality insights based on topics discussed and communication style
✓ Improved character and values assessment through expressed ideas and perspectives

### DeepSeek Schema Integration Fix (July 7, 2025)
✓ Fixed critical schema validation issue preventing DeepSeek model usage
✓ Updated all Zod schemas to include "deepseek" as valid model option
✓ Changed default model from "openai" to "deepseek" across all schemas
✓ Added DeepSeek to service status display with green indicator
✓ Resolved video upload failures when using DeepSeek model

### Enhanced Cognitive Profiling System (July 7, 2025)
✓ Implemented comprehensive cognitive assessment including:
  - Intelligence level estimation with evidence
  - Cognitive strengths and weaknesses identification
  - Processing style analysis (analytical vs intuitive)
  - Mental agility assessment

✓ Evidence-based analysis system:
  - Direct quotations from speech analysis
  - Specific visual cues and behavioral indicators
  - Observable evidence cited for all assessments

✓ Enhanced analysis prompt structure:
  - Added cognitive_profile section with detailed assessments
  - Improved speech_analysis with vocabulary analysis
  - Added visual_evidence section for facial expressions and body language
  - Enhanced behavioral_indicators for personality trait evidence

### DeepSeek Integration & LLM Expansion (July 7, 2025)
✓ Set DeepSeek as default AI model across all analysis endpoints
✓ Updated fallback chains to prioritize DeepSeek
✓ Enhanced model selection interface in frontend
✓ Maintained compatibility with OpenAI, Anthropic, and Perplexity models

### TXT Export Support (July 7, 2025)
✓ Added comprehensive TXT export functionality
✓ Implemented generateAnalysisTxt function with proper formatting
✓ Updated download route to support txt format
✓ Enhanced frontend with TXT download buttons
✓ Updated API types to include "txt" format option

### Multi-Service Fallback Architecture (Previous)
✓ Implemented robust fallback chains for all external services
✓ Enhanced transcription with multiple service support
✓ Improved facial analysis reliability with service redundancy

## User Preferences
- Default AI Model: Anthropic Claude (changed from DeepSeek per user request)
- Analysis Focus: Comprehensive cognitive and psychological profiling with speech-first analysis
- Evidence Requirement: All assessments must be supported by observable evidence
- Download Formats: PDF, Word, and TXT support required
- Service Reliability: Fallback chains preferred for all external services
- Formatting: Plain text output without markdown formatting (# and ### removed)

## Project Architecture

### Analysis Pipeline
1. **Media Processing**: Images/videos processed with facial analysis and transcription
2. **Multi-Service Analysis**: Fallback chains ensure reliability across all services
3. **Enhanced AI Analysis**: Cognitive profiling with evidence-based reasoning
4. **Document Generation**: Multiple export formats with comprehensive formatting
5. **Real-time Chat**: Follow-up questions and clarifications supported

### API Endpoints
- `/api/upload/media` - Media upload and analysis
- `/api/analyze/text` - Text analysis
- `/api/analyze/document` - Document analysis
- `/api/download/:id` - Multi-format document download
- `/api/chat` - Real-time chat interface
- `/api/share` - Analysis sharing functionality

### Frontend Components
- Media upload interface with drag-and-drop
- Model selection with real-time service status
- Analysis results with formatted display
- Download options (PDF, Word, TXT)
- Chat interface for follow-up questions
- Session management and history

## Development Notes
- Enhanced analysis prompts provide comprehensive cognitive assessments
- Evidence-based reasoning ensures scientific rigor in all analyses
- TXT export maintains formatting and structure for accessibility
- Multi-service architecture provides reliability and redundancy
- DeepSeek integration offers cost-effective high-quality analysis