import type { Express, Request, Response, NextFunction } from "express";
import { createServer, type Server } from "http";
import { storage } from "./storage";
import OpenAI from "openai";
import { insertAnalysisSchema, insertMessageSchema, insertShareSchema, uploadMediaSchema } from "@shared/schema";
import { z } from "zod";
import { 
  RekognitionClient, 
  DetectFacesCommand, 
  StartFaceDetectionCommand, 
  GetFaceDetectionCommand 
} from "@aws-sdk/client-rekognition";
import { sendAnalysisEmail } from "./services/email";
import { generateAnalysisHtml, generatePdf, generateDocx, generateAnalysisTxt } from './services/document';
import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { promisify } from 'util';
import ffmpeg from 'fluent-ffmpeg';
import Anthropic from '@anthropic-ai/sdk';
import fetch from 'node-fetch';
import FormData from 'form-data';

// Initialize API clients with proper error handling for missing keys
let openai: OpenAI | null = null;
let anthropic: Anthropic | null = null;
let azureOpenAI: OpenAI | null = null;
let deepseek: OpenAI | null = null;

// API Keys available for various services
const GLADIA_API_KEY = process.env.GLADIA_API_KEY;
const ASSEMBLYAI_API_KEY = process.env.ASSEMBLYAI_API_KEY;
const DEEPGRAM_API_KEY = process.env.DEEPGRAM_API_KEY;
const FACEPP_API_KEY = process.env.FACEPP_API_KEY;
const FACEPP_API_SECRET = process.env.FACEPP_API_SECRET;
const AZURE_FACE_API_KEY = process.env.AZURE_FACE_API_KEY;
const AZURE_FACE_ENDPOINT = process.env.AZURE_FACE_ENDPOINT;
const GOOGLE_CLOUD_VISION_API_KEY = process.env.GOOGLE_CLOUD_VISION_API_KEY;
const AZURE_VIDEO_INDEXER_KEY = process.env.AZURE_VIDEO_INDEXER_KEY;
const AZURE_VIDEO_INDEXER_LOCATION = process.env.AZURE_VIDEO_INDEXER_LOCATION;
const AZURE_VIDEO_INDEXER_ACCOUNT_ID = process.env.AZURE_VIDEO_INDEXER_ACCOUNT_ID;

// Log available APIs for transcription
if (GLADIA_API_KEY) {
  console.log("Gladia transcription API available");
}

if (ASSEMBLYAI_API_KEY) {
  console.log("AssemblyAI transcription API available");
}

if (DEEPGRAM_API_KEY) {
  console.log("Deepgram transcription API available");
}

// Availability of face analysis APIs
if (FACEPP_API_KEY && FACEPP_API_SECRET) {
  console.log("Face++ API available for face analysis");
}

if (AZURE_FACE_API_KEY && AZURE_FACE_ENDPOINT) {
  console.log("Azure Face API available for face analysis");
}

if (GOOGLE_CLOUD_VISION_API_KEY) {
  console.log("Google Cloud Vision API available for image analysis");
}

// Availability of video analysis
if (AZURE_VIDEO_INDEXER_KEY && AZURE_VIDEO_INDEXER_LOCATION && AZURE_VIDEO_INDEXER_ACCOUNT_ID) {
  console.log("Azure Video Indexer API available for deep video analysis");
}

// Deepgram API will be used via direct fetch calls instead of SDK

// Initialize Azure OpenAI if available
if (process.env.AZURE_OPENAI_KEY && process.env.AZURE_OPENAI_ENDPOINT) {
  try {
    // Initialize the Azure OpenAI client
    azureOpenAI = new OpenAI({
      apiKey: process.env.AZURE_OPENAI_KEY,
      baseURL: `${process.env.AZURE_OPENAI_ENDPOINT}/openai/deployments/gpt-4/`,
      defaultQuery: { "api-version": "2023-12-01-preview" },
      defaultHeaders: { "api-key": process.env.AZURE_OPENAI_KEY }
    });
    console.log("Azure OpenAI client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize Azure OpenAI client:", error);
  }
}

// Initialize DeepSeek client (using OpenAI-compatible API)
if (process.env.DEEPSEEK_API_KEY) {
  try {
    deepseek = new OpenAI({
      apiKey: process.env.DEEPSEEK_API_KEY,
      baseURL: "https://api.deepseek.com/v1"
    });
    console.log("DeepSeek client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize DeepSeek client:", error);
  }
}

// Check if OpenAI API key is available
if (process.env.OPENAI_API_KEY) {
  try {
    openai = new OpenAI({ apiKey: process.env.OPENAI_API_KEY });
    console.log("OpenAI client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize OpenAI client:", error);
  }
} else {
  console.warn("OPENAI_API_KEY environment variable is not set. OpenAI API functionality will be limited.");
}

// Check if Anthropic API key is available
if (process.env.ANTHROPIC_API_KEY) {
  try {
    anthropic = new Anthropic({ apiKey: process.env.ANTHROPIC_API_KEY });
    console.log("Anthropic client initialized successfully");
  } catch (error) {
    console.error("Failed to initialize Anthropic client:", error);
  }
} else {
  console.warn("ANTHROPIC_API_KEY environment variable is not set. Anthropic API functionality will be limited.");
}

// Perplexity AI client
const perplexity = {
  query: async ({ model, query }: { model: string, query: string }) => {
    if (!process.env.PERPLEXITY_API_KEY) {
      console.warn("PERPLEXITY_API_KEY environment variable is not set. Perplexity API functionality will be limited.");
      throw new Error("Perplexity API key not available");
    }
    
    try {
      const response = await fetch("https://api.perplexity.ai/chat/completions", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Bearer ${process.env.PERPLEXITY_API_KEY}`
        },
        body: JSON.stringify({
          model,
          messages: [{ role: "user", content: query }]
        })
      });
      
      const data = await response.json();
      return {
        text: data.choices[0]?.message?.content || ""
      };
    } catch (error) {
      console.error("Perplexity API error:", error);
      return { text: "" };
    }
  }
};

// AWS Rekognition client
// Let the AWS SDK pick up credentials from environment variables automatically
const rekognition = new RekognitionClient({ 
  region: process.env.AWS_REGION || "us-east-1"
});

// For Google Cloud functionality, we'll implement in a follow-up task

// For temporary file storage
const tempDir = os.tmpdir();
const writeFileAsync = promisify(fs.writeFile);
const unlinkAsync = promisify(fs.unlink);

// 50-QUESTION FRAMEWORKS FOR EACH ANALYSIS TYPE

const PHOTO_ANALYSIS_QUESTIONS = [
  // I. Physical Cues (10)
  "What is the person's approximate age range, and what visual evidence supports this?",
  "What is their likely dominant hand, based on body posture or hand use?",
  "What kind of lighting was used (natural, fluorescent, LED), and how does it shape facial tone or mood?",
  "How symmetrical is the person's face, and what asymmetries are visible?",
  "Describe the color and apparent texture of the person's skin in objective terms.",
  "Identify one visible physical trait (scar, mole, wrinkle pattern) and infer its probable significance (age, stress, lifestyle).",
  "What can be inferred about the person's sleep habits from the eyes and skin tone?",
  "Describe the person's hair (color, grooming, direction, style) and what it indicates about self-presentation.",
  "What kind of lighting shadow falls across the eyes or nose, and what mood does that lighting convey?",
  "Is there evidence of cosmetic enhancement (makeup, filters, retouching), and how does it alter authenticity?",
  
  // II. Expression & Emotion (10)
  "Describe the dominant facial expression in granular terms (eyebrow position, lip tension, gaze angle).",
  "Does the expression look posed or spontaneous? Why?",
  "Identify micro-expressions suggesting secondary emotions (e.g., contempt, anxiety, curiosity).",
  "Does the smile (if any) engage the eyes? What does that reveal psychologically?",
  "Compare upper-face emotion vs. lower-face emotion; do they match?",
  "What emotional tone is conveyed by the person's gaze direction (camera, away, downward)?",
  "Does the person appear guarded, open, or performative? Cite visible evidence.",
  "Are there tension points in the jaw or neck suggesting repressed emotion?",
  "Estimate how long the expression was held for the photo.",
  "Does the emotion appear congruent with the setting or mismatched? What does that mismatch suggest?",
  
  // III. Composition & Context (10)
  "Describe the setting (indoor/outdoor, professional/personal) and how it relates to self-presentation.",
  "What objects or background details signal aspects of lifestyle or occupation?",
  "How does clothing color palette interact with lighting to create an emotional tone?",
  "What focal length or camera distance was likely used, and how does it affect psychological intimacy?",
  "Is there visible clutter or minimalism, and what does that suggest about personality?",
  "Are there reflections, windows, or mirrors in frame? What might they symbolize?",
  "How does body posture interact with spatial framing (e.g., leaning toward/away from camera)?",
  "What portion of the frame the subject occupies, and what does that say about ego strength or humility?",
  "Is there visible symmetry or imbalance in composition, and what does it communicate?",
  "Identify one hidden or easily overlooked element that subtly changes the psychological reading.",
  
  // IV. Personality & Psychological Inference (10)
  "Based on facial micro-cues, what is the person's baseline affect (calm, anxious, irritable, contemplative)?",
  "What defense mechanism seems most active (projection, reaction formation, displacement, denial)?",
  "Describe the likely self-image being projected—how do posture, expression, and clothing support it?",
  "What aspects of the photo seem unconsciously revealing versus deliberately controlled?",
  "How would the person handle confrontation, judging by visible muscular tension or gaze stability?",
  "Does the person exhibit signs of narcissism or self-doubt? Identify visible indicators.",
  "What cognitive style is implied (systematic, intuitive, chaotic)?",
  "What is the person's apparent relationship to vulnerability? Cite visual evidence.",
  "Does the photo suggest recent emotional hardship or resilience?",
  "How does the person seem to want to be seen—and what discrepancy exists between that and how they actually appear?",
  
  // V. Symbolic & Metapsychological Analysis (10)
  "What emotional temperature (warm/cool) dominates the photo's color space, and what archetype does it evoke?",
  "If the photo were a dream image, what would each major element (pose, setting, color) symbolize?",
  "What mythic or cinematic archetype does the person most resemble, and why?",
  "Which aspect of the psyche (persona, shadow, anima/animus, self) is most visible?",
  "What unconscious conflict seems dramatized in the composition?",
  "How does the person's clothing or accessories function as psychological armor?",
  "What is the implied relationship between the photographer and subject (trust, tension, dominance)?",
  "If this image were part of a sequence, what emotional narrative would it tell?",
  "What single object or feature in the photo best symbolizes the person's life stance?",
  "What inner contradiction or paradox defines the subject, as revealed through visible cues?"
];

const VIDEO_ANALYSIS_QUESTIONS = [
  // I. Physical & Behavioral Cues (10)
  "How does the person's gait or movement rhythm change across the clip?",
  "Which recurring gesture seems habitual rather than situational?",
  "Describe one moment where muscle tension releases or spikes — what triggers it?",
  "How does posture vary when the person speaks vs. listens?",
  "Identify one micro-adjustment (e.g., hair touch, collar fix) and explain its likely emotional cause.",
  "What is the person doing with their hands during silent intervals?",
  "How consistent is eye-contact across frames? Give timestamps showing breaks or sustained gazes.",
  "At which point does breathing rate visibly change, and what precedes it?",
  "Describe the physical energy level throughout — rising, falling, or cyclical?",
  "What body part seems most expressive (eyes, shoulders, mouth), and how is that used?",
  
  // II. Expression & Emotion Over Time (10)
  "Track micro-expressions that flicker and vanish. At what timestamps do they appear?",
  "When does the dominant emotion shift, and how abruptly?",
  "Does the person's smile fade naturally or snap off?",
  "Which emotion seems performed vs. spontaneous? Cite frames.",
  "How does blink rate change when discussing specific topics?",
  "Identify one involuntary facial tic and interpret its significance.",
  "Are there moments of incongruence between facial expression and vocal tone?",
  "When does the person's face 'freeze' — i.e., hold still unnaturally — and what triggers that?",
  "What subtle expression signals discomfort before any verbal cue?",
  "How does lighting or camera angle amplify or mute visible emotions?",
  
  // III. Speech, Voice & Timing (10)
  "Describe baseline vocal timbre — breathy, clipped, resonant — and what personality trait it implies.",
  "At which timestamp does pitch spike or flatten dramatically? Why?",
  "How does speaking rate change when emotionally charged content arises?",
  "Identify one pause longer than 1.5 seconds and interpret it psychologically.",
  "What filler words or vocal tics recur, and what function do they serve?",
  "How synchronized are gestures with speech rhythm?",
  "Does the voice carry underlying fatigue, tension, or confidence? Provide audible markers.",
  "Compare early vs. late segments: does articulation become more or less precise?",
  "What is the emotional contour of the voice across the clip (anxious → calm, etc.)?",
  "When does volume drop below baseline, and what coincides with it visually?",
  
  // IV. Context, Environment & Interaction (10)
  "What environmental cues (background noise, lighting shifts) change mid-video?",
  "How does the camera distance or angle influence perceived dominance or submission?",
  "Are there off-screen sounds or glances suggesting another presence?",
  "When the person looks away, where do they look, and what might they be avoiding?",
  "How do objects in the frame get used or ignored (cup, pen, phone)?",
  "Does the person adapt posture or tone in response to environmental change (light flicker, sound)?",
  "What part of the environment most reflects personality (book titles, wall art, tidiness)?",
  "How does background color palette influence mood perception?",
  "Is there evidence of editing cuts or jump transitions that alter authenticity?",
  "What temporal pacing (camera motion, cut frequency) matches or mismatches the emotional tempo?",
  
  // V. Personality & Psychological Inference (10)
  "Based on kinetic patterns, what baseline temperament (introvert/extrovert, restrained/expressive) emerges?",
  "What defense mechanism manifests dynamically (e.g., laughter after stress cue)?",
  "When does self-presentation collapse momentarily into candor?",
  "What behavioral marker suggests anxiety management (fidgeting, throat clearing, leg bounce)?",
  "How does the person handle silence — restless, composed, avoidant?",
  "Identify one moment that feels genuinely unguarded; what detail proves it?",
  "What relational stance is enacted toward the viewer (teacher, confessor, performer)?",
  "Does the body ever contradict the words? Provide timestamps.",
  "What sustained pattern (voice-tone loop, repeated motion) indicates underlying psychological theme?",
  "What overall transformation occurs from first to last frame — and what emotional or existential story does that evolution tell?"
];

const TEXT_ANALYSIS_QUESTIONS = [
  // I. Language & Style (10)
  "What is the dominant sentence rhythm — clipped, flowing, erratic — and what personality trait does it reveal?",
  "Which adjectives recur, and what emotional bias do they show?",
  "How does pronoun use ('I,' 'you,' 'we,' 'they') shift across the text, and what identity stance does that reflect?",
  "What level of abstraction vs. concreteness dominates the writing?",
  "Identify one passage where diction becomes suddenly elevated or deflated — what triggers it?",
  "Are there unfinished or fragmentary sentences, and what might that signal psychologically?",
  "How consistent is the tense? Does the writer slip between past and present, and why?",
  "What metaphors or analogies recur, and what unconscious associations do they expose?",
  "Is the author's tone self-assured, tentative, ironic, or performative? Cite phrasing.",
  "What linguistic register (formal, colloquial, technical) dominates, and how does it align with self-image?",
  
  // II. Emotional Indicators (10)
  "What emotion seems primary (anger, melancholy, pride, longing), and where is it linguistically concentrated?",
  "Which emotions appear repressed or displaced — hinted at but never named?",
  "Does emotional intensity rise or fall as the text progresses?",
  "Identify one sentence where affect 'leaks through' despite apparent control.",
  "Are there moments of sentimental overstatement or cold detachment?",
  "What bodily or sensory words appear, and what do they suggest about embodiment or repression?",
  "Is there ambivalence toward the subject matter? Cite a line where tone wavers.",
  "Does humor appear, and if so, is it self-directed, aggressive, or defensive?",
  "What words betray anxiety or guilt?",
  "How is desire represented — directly, symbolically, or through avoidance?",
  
  // III. Cognitive & Structural Patterns (10)
  "How logically coherent are transitions between ideas?",
  "Does the writer prefer enumeration, narrative, or digression? What does that indicate about thought style?",
  "What syntactic habits dominate (parallelism, repetition, parenthesis), and what mental rhythms do they mirror?",
  "Are there contradictions the author fails to notice? Quote one.",
  "How does the author handle uncertainty — through hedging, assertion, or silence?",
  "Does the argument or story circle back on itself?",
  "Are there abrupt topic shifts, and what emotional events coincide with them?",
  "What elements of the text seem compulsive or ritualistic in repetition?",
  "Where does the writer show real insight versus mechanical reasoning?",
  "How does closure occur (resolution, withdrawal, collapse), and what does it signify psychologically?",
  
  // IV. Self-Representation & Identity (10)
  "How does the writer portray the self — victim, hero, observer, analyst?",
  "Is there a split between narrating voice and lived experience?",
  "What form of authority or validation does the author seek (moral, intellectual, emotional)?",
  "How consistent is the self-image across paragraphs?",
  "Identify one phrase that reveals unconscious self-evaluation (admiration, contempt, shame).",
  "Does the author reveal dependency on external approval or autonomy from it?",
  "What form of vulnerability does the writer allow?",
  "How does the author talk about others — with empathy, rivalry, indifference?",
  "What implicit audience is being addressed?",
  "Does the writer's stance shift from confession to performance? Cite turning point.",
  
  // V. Symbolic & Unconscious Material (10)
  "Which images or motifs recur (light/dark, ascent/descent, enclosure, mirrors), and what do they symbolize?",
  "Are there dream-like or surreal elements?",
  "What oppositions structure the text (order/chaos, love/power, mind/body)?",
  "What wish or fear seems to animate the text beneath the surface argument?",
  "Identify one metaphor that reads like a disguised confession.",
  "How does the author relate to time — nostalgic, future-oriented, frozen?",
  "Does the text express conflict between intellect and emotion?",
  "What shadow aspect of personality is hinted at through hostile or taboo imagery?",
  "Is there evidence of projection — attributing inner states to others or to abstractions?",
  "What central psychological drama (loss, control, recognition, transformation) structures the entire piece?"
];

// Helper function to clean markdown formatting from analysis text
function cleanMarkdownFromAnalysis(obj: any): any {
  if (typeof obj === 'string') {
    return obj
      .replace(/#{1,6}\s*/g, '') // Remove markdown headers
      .replace(/\*\*(.*?)\*\*/g, '$1') // Remove bold formatting
      .replace(/\*(.*?)\*/g, '$1') // Remove italic formatting
      .replace(/```[\s\S]*?```/g, '') // Remove code blocks
      .replace(/`([^`]+)`/g, '$1') // Remove inline code
      .trim();
  } else if (Array.isArray(obj)) {
    return obj.map(item => cleanMarkdownFromAnalysis(item));
  } else if (obj && typeof obj === 'object') {
    const cleaned: any = {};
    for (const [key, value] of Object.entries(obj)) {
      cleaned[key] = cleanMarkdownFromAnalysis(value);
    }
    return cleaned;
  }
  return obj;
}

// Google Cloud Storage bucket for videos
// This would typically be created and configured through Google Cloud Console first

/**
 * Comprehensive multi-service face analysis using ALL available services
 * Integrates Google Cloud Vision, Face++, Azure Face, and AWS Rekognition
 */
async function comprehensiveMultiServiceFaceAnalysis(imageBuffer: Buffer, maxPeople: number = 5): Promise<any[]> {
  console.log('Starting comprehensive multi-service face analysis...');
  
  const analysisResults: any = {
    azure_face: null,
    facepp: null,
    google_vision: null,
    aws_rekognition: null,
    people_detected: 0
  };
  
  // Run ALL services in parallel for maximum data collection
  const servicePromises = [];
  
  // 1. Azure Face API Analysis
  if (AZURE_FACE_API_KEY && AZURE_FACE_ENDPOINT) {
    servicePromises.push(
      (async () => {
        try {
          console.log('Attempting face analysis with Azure Face API...');
          const response = await fetch(
            `${AZURE_FACE_ENDPOINT}/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true&returnFaceAttributes=age,gender,headPose,smile,facialHair,glasses,emotion,hair,makeup,occlusion,accessories,blur,exposure,noise`,
            {
              method: 'POST',
              headers: {
                'Ocp-Apim-Subscription-Key': AZURE_FACE_API_KEY,
                'Content-Type': 'application/octet-stream'
              },
              body: imageBuffer
            }
          );
          
          if (response.ok) {
            const data = await response.json() as any;
            analysisResults.azure_face = data;
            console.log('Azure Face API analysis successful!');
          } else {
            const errorText = await response.text();
            console.log('Azure Face API returned an error:', errorText);
            analysisResults.azure_face = { error: errorText };
          }
        } catch (error: any) {
          console.error('Azure Face API error:', error);
          analysisResults.azure_face = { error: error.message };
        }
      })()
    );
  }
  
  // 2. Face++ API Analysis
  if (FACEPP_API_KEY && FACEPP_API_SECRET) {
    servicePromises.push(
      (async () => {
        try {
          console.log('Attempting face analysis with Face++ API...');
          const formData = new FormData();
          formData.append('api_key', FACEPP_API_KEY);
          formData.append('api_secret', FACEPP_API_SECRET);
          formData.append('image_base64', imageBuffer.toString('base64'));
          formData.append('return_attributes', 'gender,age,smiling,headpose,facequality,blur,eyestatus,emotion,ethnicity,beauty,mouthstatus,eyegaze,skinstatus');
          
          const response = await fetch('https://api-us.faceplusplus.com/facepp/v3/detect', {
            method: 'POST',
            body: formData
          });
          
          if (response.ok) {
            const data = await response.json() as any;
            analysisResults.facepp = data;
            console.log('Face++ API analysis successful!');
          } else {
            const errorText = await response.text();
            console.log('Face++ API returned an error:', errorText);
            analysisResults.facepp = { error: errorText };
          }
        } catch (error: any) {
          console.error('Face++ API error:', error);
          analysisResults.facepp = { error: error.message };
        }
      })()
    );
  }
  
  // 3. Google Cloud Vision API Analysis
  if (GOOGLE_CLOUD_VISION_API_KEY) {
    servicePromises.push(
      (async () => {
        try {
          console.log('Attempting face analysis with Google Cloud Vision API...');
          const requestBody = {
            requests: [{
              image: {
                content: imageBuffer.toString('base64')
              },
              features: [
                { type: 'FACE_DETECTION', maxResults: maxPeople },
                { type: 'OBJECT_LOCALIZATION', maxResults: 10 },
                { type: 'TEXT_DETECTION', maxResults: 5 },
                { type: 'LABEL_DETECTION', maxResults: 10 }
              ]
            }]
          };
          
          const response = await fetch(
            `https://vision.googleapis.com/v1/images:annotate?key=${GOOGLE_CLOUD_VISION_API_KEY}`,
            {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify(requestBody)
            }
          );
          
          if (response.ok) {
            const data = await response.json() as any;
            analysisResults.google_vision = data.responses[0];
            console.log('Google Cloud Vision API analysis successful!');
          } else {
            const errorText = await response.text();
            console.log('Google Cloud Vision API returned an error:', errorText);
            analysisResults.google_vision = { error: errorText };
          }
        } catch (error: any) {
          console.error('Google Cloud Vision API error:', error);
          analysisResults.google_vision = { error: error.message };
        }
      })()
    );
  }
  
  // 4. AWS Rekognition Analysis (fallback)
  servicePromises.push(
    (async () => {
      try {
        console.log('Attempting face analysis with AWS Rekognition...');
        analysisResults.aws_rekognition = await analyzeFaceWithRekognition(imageBuffer, maxPeople);
        console.log('AWS Rekognition analysis successful!');
      } catch (error: any) {
        console.error('AWS Rekognition error:', error);
        analysisResults.aws_rekognition = { error: error.message };
      }
    })()
  );
  
  // Wait for all services to complete
  await Promise.all(servicePromises);
  
  // Process and integrate results from all services
  const integratedResults = integrateMultiServiceResults(analysisResults, maxPeople);
  
  console.log(`Comprehensive analysis complete. Integrated data from ${Object.keys(analysisResults).filter(key => analysisResults[key] && !analysisResults[key].error).length} services.`);
  
  return integratedResults;
}

/**
 * Integrate results from multiple face analysis services into comprehensive profiles
 */
function integrateMultiServiceResults(analysisResults: any, maxPeople: number): any[] {
  const integratedPeople: any[] = [];
  
  // Start with Face++ results as primary (most comprehensive)
  if (analysisResults.facepp && analysisResults.facepp.faces && analysisResults.facepp.faces.length > 0) {
    for (let i = 0; i < Math.min(analysisResults.facepp.faces.length, maxPeople); i++) {
      const face = analysisResults.facepp.faces[i];
      const person = {
        personIndex: i + 1,
        personLabel: `Person ${i + 1} (Male)`, // Will be updated based on analysis
        boundingBox: face.face_rectangle,
        
        // Comprehensive multi-service data
        multiServiceData: {
          facepp: face,
          azure_face: null,
          google_vision: null,
          aws_rekognition: null
        },
        
        // Integrated analysis
        integratedAnalysis: {
          age: face.attributes?.age?.value || null,
          gender: face.attributes?.gender?.value || 'unknown',
          emotions: {
            primary: getTopEmotion(face.attributes?.emotion),
            detailed: face.attributes?.emotion || {}
          },
          facial_features: {
            smiling: face.attributes?.smiling?.value || 0,
            glasses: face.attributes?.glass?.value || 'none',
            facial_hair: face.attributes?.beard?.value || 0,
            ethnicity: face.attributes?.ethnicity?.value || 'unknown'
          },
          psychological_indicators: {
            confidence_level: face.attributes?.beauty?.male_score || face.attributes?.beauty?.female_score || 50,
            emotional_stability: calculateEmotionalStability(face.attributes?.emotion),
            social_openness: face.attributes?.smiling?.value || 0
          }
        }
      };
      
      // Update gender in label
      if (person.integratedAnalysis.gender) {
        person.personLabel = `Person ${i + 1} (${person.integratedAnalysis.gender === 'Male' ? 'Male' : 'Female'})`;
      }
      
      integratedPeople.push(person);
    }
  }
  
  // Enhance with Azure Face data if available
  if (analysisResults.azure_face && Array.isArray(analysisResults.azure_face) && analysisResults.azure_face.length > 0) {
    for (let i = 0; i < Math.min(analysisResults.azure_face.length, integratedPeople.length); i++) {
      const azureFace = analysisResults.azure_face[i];
      if (integratedPeople[i]) {
        integratedPeople[i].multiServiceData.azure_face = azureFace;
        
        // Enhance with Azure-specific data
        if (azureFace.faceAttributes) {
          integratedPeople[i].integratedAnalysis.azure_insights = {
            age: azureFace.faceAttributes.age,
            emotion: azureFace.faceAttributes.emotion,
            facial_hair: azureFace.faceAttributes.facialHair,
            glasses: azureFace.faceAttributes.glasses,
            makeup: azureFace.faceAttributes.makeup,
            accessories: azureFace.faceAttributes.accessories
          };
        }
      }
    }
  }
  
  // Enhance with Google Vision data if available
  if (analysisResults.google_vision && analysisResults.google_vision.faceAnnotations) {
    for (let i = 0; i < Math.min(analysisResults.google_vision.faceAnnotations.length, integratedPeople.length); i++) {
      const googleFace = analysisResults.google_vision.faceAnnotations[i];
      if (integratedPeople[i]) {
        integratedPeople[i].multiServiceData.google_vision = googleFace;
        
        // Add Google-specific insights
        integratedPeople[i].integratedAnalysis.google_insights = {
          joy_likelihood: googleFace.joyLikelihood,
          sorrow_likelihood: googleFace.sorrowLikelihood,
          anger_likelihood: googleFace.angerLikelihood,
          surprise_likelihood: googleFace.surpriseLikelihood,
          detection_confidence: googleFace.detectionConfidence,
          landmarks: googleFace.landmarks?.length || 0
        };
      }
    }
  }
  
  // If no Face++ results, fall back to other services
  if (integratedPeople.length === 0) {
    // Try Azure Face as fallback
    if (analysisResults.azure_face && Array.isArray(analysisResults.azure_face) && analysisResults.azure_face.length > 0) {
      for (let i = 0; i < Math.min(analysisResults.azure_face.length, maxPeople); i++) {
        const azureFace = analysisResults.azure_face[i];
        integratedPeople.push({
          personIndex: i + 1,
          personLabel: `Person ${i + 1} (${azureFace.faceAttributes?.gender || 'Unknown'})`,
          boundingBox: azureFace.faceRectangle,
          multiServiceData: { azure_face: azureFace, facepp: null, google_vision: null, aws_rekognition: null },
          integratedAnalysis: {
            age: azureFace.faceAttributes?.age,
            gender: azureFace.faceAttributes?.gender,
            emotions: { primary: getTopEmotionFromAzure(azureFace.faceAttributes?.emotion), detailed: azureFace.faceAttributes?.emotion },
            psychological_indicators: { confidence_level: 70, emotional_stability: 50, social_openness: azureFace.faceAttributes?.smile || 0 }
          }
        });
      }
    }
    // Try Google Vision as final fallback
    else if (analysisResults.google_vision && analysisResults.google_vision.faceAnnotations) {
      for (let i = 0; i < Math.min(analysisResults.google_vision.faceAnnotations.length, maxPeople); i++) {
        const googleFace = analysisResults.google_vision.faceAnnotations[i];
        integratedPeople.push({
          personIndex: i + 1,
          personLabel: `Person ${i + 1}`,
          boundingBox: googleFace.boundingPoly,
          multiServiceData: { google_vision: googleFace, facepp: null, azure_face: null, aws_rekognition: null },
          integratedAnalysis: {
            emotions: { primary: googleFace.joyLikelihood, detailed: { joy: googleFace.joyLikelihood, sorrow: googleFace.sorrowLikelihood } },
            psychological_indicators: { confidence_level: googleFace.detectionConfidence * 100, emotional_stability: 50, social_openness: 50 }
          }
        });
      }
    }
  }
  
  // Add comprehensive service status
  integratedPeople.forEach(person => {
    person.serviceStatus = {
      facepp_available: !!(analysisResults.facepp && !analysisResults.facepp.error),
      azure_face_available: !!(analysisResults.azure_face && !analysisResults.azure_face.error),
      google_vision_available: !!(analysisResults.google_vision && !analysisResults.google_vision.error),
      aws_rekognition_available: !!(analysisResults.aws_rekognition && !analysisResults.aws_rekognition.error),
      total_services_used: Object.keys(analysisResults).filter(key => analysisResults[key] && !analysisResults[key].error).length
    };
  });
  
  return integratedPeople;
}

/**
 * Helper function to get top emotion from Face++ results
 */
function getTopEmotion(emotions: any): string {
  if (!emotions) return 'neutral';
  
  let topEmotion = 'neutral';
  let maxValue = 0;
  
  Object.keys(emotions).forEach(emotion => {
    if (emotions[emotion] > maxValue) {
      maxValue = emotions[emotion];
      topEmotion = emotion;
    }
  });
  
  return topEmotion;
}

/**
 * Helper function to get top emotion from Azure Face results
 */
function getTopEmotionFromAzure(emotions: any): string {
  if (!emotions) return 'neutral';
  
  let topEmotion = 'neutral';
  let maxValue = 0;
  
  Object.keys(emotions).forEach(emotion => {
    if (emotions[emotion] > maxValue) {
      maxValue = emotions[emotion];
      topEmotion = emotion;
    }
  });
  
  return topEmotion;
}

/**
 * Calculate emotional stability from emotion scores
 */
function calculateEmotionalStability(emotions: any): number {
  if (!emotions) return 50;
  
  const emotionValues = Object.values(emotions) as number[];
  const variance = emotionValues.reduce((sum, val) => sum + Math.pow(val - 50, 2), 0) / emotionValues.length;
  return Math.max(0, Math.min(100, 100 - variance));
}
const bucketName = 'ai-personality-videos';

/**
 * Helper function to get the duration of a video using ffprobe
 */
async function getVideoDuration(videoPath: string): Promise<number> {
  return new Promise<number>((resolve, reject) => {
    try {
      ffmpeg.ffprobe(videoPath, (err: Error | null, metadata: any) => {
        if (err) {
          console.error('Error getting video duration:', err);
          console.error('ffprobe error details:', err.message);
          // Default to 5 seconds if we can't determine duration
          return resolve(5);
        }
        
        // Get duration in seconds
        const durationSec = metadata.format.duration || 5;
        console.log(`Video duration detected: ${durationSec}s`);
        resolve(durationSec);
      });
    } catch (error) {
      console.error('ffprobe not available or failed:', error);
      // Fallback to default duration if ffmpeg is not available
      resolve(5);
    }
  });
}

/**
 * Helper function to split a video into chunks of specified duration
 */
async function splitVideoIntoChunks(videoPath: string, outputDir: string, chunkDurationSec: number): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    ffmpeg(videoPath)
      .outputOptions([
        `-f segment`,
        `-segment_time ${chunkDurationSec}`,
        `-reset_timestamps 1`,
        `-c copy` // Copy codec (fast)
      ])
      .output(path.join(outputDir, 'chunk_%03d.mp4'))
      .on('end', () => {
        console.log('Video successfully split into chunks');
        resolve();
      })
      .on('error', (err: Error) => {
        console.error('Error splitting video:', err);
        reject(err);
      })
      .run();
  });
}

/**
 * Helper function to extract a specific 3-second segment from a video
 */
async function extractVideoSegment(videoPath: string, startTime: number, duration: number, outputPath: string): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    ffmpeg(videoPath)
      .seekInput(startTime)
      .duration(duration)
      .outputOptions(['-c:v libx264', '-c:a aac'])
      .output(outputPath)
      .on('end', () => {
        console.log(`Video segment extracted: ${startTime}s to ${startTime + duration}s`);
        resolve();
      })
      .on('error', (err: Error) => {
        console.error('Error extracting video segment:', err);
        reject(err);
      })
      .run();
  });
}

/**
 * Helper function to analyze video using Azure Video Indexer
 * Extracts insights about scenes, emotions, and content
 */
async function analyzeVideoWithAzureIndexer(videoBuffer: Buffer): Promise<any> {
  // Check if Azure Video Indexer keys are available
  if (!AZURE_VIDEO_INDEXER_KEY || !AZURE_VIDEO_INDEXER_LOCATION || !AZURE_VIDEO_INDEXER_ACCOUNT_ID) {
    console.warn('Azure Video Indexer credentials not available');
    return null;
  }
  
  try {
    console.log('Starting Azure Video Indexer analysis...');
    
    // Step 1: Get an access token for the Video Indexer API
    const accessTokenResponse = await fetch(
      `https://api.videoindexer.ai/auth/${AZURE_VIDEO_INDEXER_LOCATION}/Accounts/${AZURE_VIDEO_INDEXER_ACCOUNT_ID}/AccessToken?allowEdit=true`,
      {
        method: 'GET',
        headers: {
          'Ocp-Apim-Subscription-Key': AZURE_VIDEO_INDEXER_KEY
        }
      }
    );
    
    if (!accessTokenResponse.ok) {
      throw new Error(`Failed to get access token: ${await accessTokenResponse.text()}`);
    }
    
    const accessToken = await accessTokenResponse.text();
    console.log('Obtained Azure Video Indexer access token');
    
    // Step 2: Create a random ID for the video
    const videoId = `video_${Date.now()}_${Math.random().toString(36).substring(2, 8)}`;
    
    // Step 3: Upload the video
    // Create form data with the video buffer
    const formData = new FormData();
    formData.append('file', videoBuffer, 'video.mp4');
    
    // @ts-ignore: FormData is compatible with fetch API's Body type
    const uploadResponse = await fetch(
      `https://api.videoindexer.ai/${AZURE_VIDEO_INDEXER_LOCATION}/Accounts/${AZURE_VIDEO_INDEXER_ACCOUNT_ID}/Videos?accessToken=${accessToken}&name=${videoId}&privacy=private&indexingPreset=Default`,
      {
        method: 'POST',
        body: formData
      }
    );
    
    if (!uploadResponse.ok) {
      throw new Error(`Failed to upload video: ${await uploadResponse.text()}`);
    }
    
    const uploadResult = await uploadResponse.json();
    const indexingVideoId = uploadResult.id;
    console.log(`Video uploaded, ID: ${indexingVideoId}`);
    
    // Step 4: Wait for indexing to complete
    let isIndexingComplete = false;
    let indexingState = "";
    let indexingRetries = 0;
    const maxRetries = 20; // Maximum number of retries
    
    while (!isIndexingComplete && indexingRetries < maxRetries) {
      await new Promise(resolve => setTimeout(resolve, 5000)); // Wait 5 seconds between checks
      
      const indexingStateResponse = await fetch(
        `https://api.videoindexer.ai/${AZURE_VIDEO_INDEXER_LOCATION}/Accounts/${AZURE_VIDEO_INDEXER_ACCOUNT_ID}/Videos/${indexingVideoId}/Index?accessToken=${accessToken}`,
        {
          method: 'GET'
        }
      );
      
      if (indexingStateResponse.ok) {
        const indexingData = await indexingStateResponse.json();
        indexingState = indexingData.state;
        
        if (indexingState === "Processed") {
          isIndexingComplete = true;
          console.log('Video indexing completed successfully');
          
          // Step 5: Get the insights from the video
          // The indexingData already contains all the insights
          
          // Step 6: Clean up - delete the video from Azure
          try {
            await fetch(
              `https://api.videoindexer.ai/${AZURE_VIDEO_INDEXER_LOCATION}/Accounts/${AZURE_VIDEO_INDEXER_ACCOUNT_ID}/Videos/${indexingVideoId}?accessToken=${accessToken}`,
              {
                method: 'DELETE'
              }
            );
            console.log('Video deleted from Azure Video Indexer');
          } catch (deleteError) {
            console.warn('Failed to delete video from Azure Video Indexer:', deleteError);
          }
          
          // Process and return the insights
          return processVideoIndexerResults(indexingData);
        } else if (indexingState === "Failed") {
          throw new Error("Video indexing failed on Azure Video Indexer");
        } else {
          console.log(`Indexing in progress, state: ${indexingState}`);
        }
      } else {
        console.warn(`Failed to get indexing state: ${await indexingStateResponse.text()}`);
      }
      
      indexingRetries++;
    }
    
    if (!isIndexingComplete) {
      throw new Error(`Video indexing timed out after ${maxRetries} retries, last state: ${indexingState}`);
    }
    
  } catch (error) {
    console.error('Azure Video Indexer analysis error:', error);
    return null;
  }
}

/**
 * Helper function to process Azure Video Indexer results
 */
function processVideoIndexerResults(indexingData: any): any {
  // Extract the most useful information from the indexing data
  const videoInsights = {
    provider: "azure_video_indexer",
    duration: indexingData.summarizedInsights?.duration || 0,
    
    // Scene analysis
    scenes: indexingData.summarizedInsights?.scenes?.map((scene: any) => ({
      id: scene.id,
      instances: scene.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end
      }))
    })) || [],
    
    // Emotion analysis
    emotions: indexingData.summarizedInsights?.emotions?.map((emotion: any) => ({
      type: emotion.type,
      instances: emotion.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end,
        confidence: instance.confidence
      }))
    })) || [],
    
    // Face detection and tracking
    faces: indexingData.summarizedInsights?.faces?.map((face: any) => ({
      id: face.id,
      name: face.name || "Unknown person",
      confidence: face.confidence,
      instances: face.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end,
        thumbnailId: instance.thumbnailId
      }))
    })) || [],
    
    // Keywords/topics
    topics: indexingData.summarizedInsights?.topics?.map((topic: any) => ({
      name: topic.name,
      confidence: topic.confidence,
      instances: topic.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end
      }))
    })) || [],
    
    // Labels (objects, actions)
    labels: indexingData.summarizedInsights?.labels?.map((label: any) => ({
      name: label.name,
      confidence: label.confidence,
      instances: label.instances?.map((instance: any) => ({
        start: instance.start,
        end: instance.end
      }))
    })) || []
  };
  
  return videoInsights;
}

/**
 * Helper function to extract audio from video and transcribe it using multiple transcription services
 * Uses Gladia (primary), AssemblyAI (secondary with emotion tagging), or Deepgram (fallback)
 */
async function extractAudioTranscription(videoPath: string): Promise<any> {
  try {
    // Extract audio from video
    const randomId = Math.random().toString(36).substring(2, 15);
    const audioPath = path.join(tempDir, `${randomId}.mp3`);
    
    console.log('Extracting audio from video...');
    await new Promise<void>((resolve, reject) => {
      ffmpeg(videoPath)
        .output(audioPath)
        .audioCodec('libmp3lame')
        .audioChannels(1)
        .audioFrequency(16000)
        .on('end', () => resolve())
        .on('error', (err: Error) => {
          console.error('Error extracting audio:', err);
          reject(err);
        })
        .run();
    });
    
    console.log('Audio extraction complete, starting transcription...');
    
    // Get the audio file details
    const audioFile = fs.createReadStream(audioPath);
    const audioBuffer = await fs.promises.readFile(audioPath);
    const audioBase64 = audioBuffer.toString('base64');
    const audioDuration = await getVideoDuration(audioPath);
    
    // Initialize results object
    let transcriptionResult: any = {
      transcription: "",
      provider: "none",
      emotion: null,
      confidence: 0,
      wordLevelData: false,
      segments: []
    };
    
    // Try Gladia API first (primary transcription service)
    if (GLADIA_API_KEY) {
      try {
        console.log('Attempting transcription with Gladia API...');
        const formData = new FormData();
        formData.append('audio', audioBuffer, 'audio.mp3');
        
        const gladiaResponse = await fetch('https://api.gladia.io/v2/transcription', {
          method: 'POST',
          headers: {
            'x-gladia-key': GLADIA_API_KEY,
          },
          // @ts-ignore: FormData is compatible with fetch API's Body type
          body: formData
        });
        
        if (gladiaResponse.ok) {
          const result = await gladiaResponse.json();
          
          if (result.prediction && result.prediction.transcription) {
            // Process utterances from Gladia segments if available
            const utterances = [];
            if (result.prediction.utterances && result.prediction.utterances.length > 0) {
              for (const utterance of result.prediction.utterances) {
                utterances.push({
                  text: utterance.text,
                  start: utterance.start,
                  end: utterance.end,
                  sentiment: "unknown" // Gladia doesn't provide sentiment
                });
              }
            } else if (result.prediction.segments && result.prediction.segments.length > 0) {
              // Use segments as utterances if utterances aren't available
              for (const segment of result.prediction.segments) {
                utterances.push({
                  text: segment.text,
                  start: segment.start,
                  end: segment.end,
                  sentiment: "unknown"
                });
              }
            } else {
              // If no segments or utterances, create a single utterance
              utterances.push({
                text: result.prediction.transcription,
                start: 0,
                end: audioDuration || 0,
                sentiment: "unknown"
              });
            }
            
            // Process words if available
            const words = [];
            if (result.prediction.words && result.prediction.words.length > 0) {
              for (const word of result.prediction.words) {
                words.push({
                  text: word.word || word.text,
                  start: word.start,
                  end: word.end,
                  confidence: word.confidence || 0.9
                });
              }
            }
            
            transcriptionResult = {
              text: result.prediction.transcription,
              provider: "gladia",
              confidence: result.prediction.confidence || 0.9,
              wordLevelData: true,
              // Store in standardized format for easy quotation extraction
              transcription: {
                full_text: result.prediction.transcription,
                utterances: utterances,
                words: words
              },
              segments: result.prediction.words || []
            };
            console.log('Gladia transcription successful!');
          }
        } else {
          console.warn('Gladia API returned non-OK response:', gladiaResponse.status);
        }
      } catch (error) {
        console.error('Gladia transcription error:', error);
      }
    }
    
    // If Gladia fails, try AssemblyAI (which adds emotion detection)
    if (transcriptionResult.provider === "none" && ASSEMBLYAI_API_KEY) {
      try {
        console.log('Attempting transcription with AssemblyAI...');
        
        // First upload the audio file
        const uploadResponse = await fetch('https://api.assemblyai.com/v2/upload', {
          method: 'POST',
          headers: {
            'Authorization': ASSEMBLYAI_API_KEY,
            'Content-Type': 'application/json'
          },
          body: audioBuffer
        });
        
        if (uploadResponse.ok) {
          const { upload_url } = await uploadResponse.json();
          
          // Submit for transcription with sentiment analysis
          const transcribeResponse = await fetch('https://api.assemblyai.com/v2/transcript', {
            method: 'POST',
            headers: {
              'Authorization': ASSEMBLYAI_API_KEY,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              audio_url: upload_url,
              sentiment_analysis: true, // Enable sentiment analysis
              entity_detection: true,   // Identify entities
              iab_categories: true      // Topic detection
            })
          });
          
          if (transcribeResponse.ok) {
            const { id } = await transcribeResponse.json();
            
            // Poll for completion (AssemblyAI is async)
            let transcript;
            let completed = false;
            
            for (let i = 0; i < 30 && !completed; i++) { // Try up to 30 times (30 seconds)
              await new Promise(resolve => setTimeout(resolve, 1000)); // Wait 1 second
              
              const pollingResponse = await fetch(`https://api.assemblyai.com/v2/transcript/${id}`, {
                headers: { 'Authorization': ASSEMBLYAI_API_KEY }
              });
              
              if (pollingResponse.ok) {
                transcript = await pollingResponse.json();
                if (transcript.status === 'completed') {
                  completed = true;
                } else if (transcript.status === 'error') {
                  console.error('AssemblyAI transcription error:', transcript.error);
                  break;
                }
              }
            }
            
            if (completed && transcript) {
              // Extract emotion data from sentiment analysis
              const emotions = transcript.sentiment_analysis_results || [];
              
              // Process the sentiment analysis into utterance segments
              const utterances = [];
              if (emotions && emotions.length > 0) {
                for (const segment of emotions) {
                  utterances.push({
                    text: segment.text,
                    start: segment.start / 1000, // Convert to seconds
                    end: segment.end / 1000,     // Convert to seconds
                    sentiment: segment.sentiment
                  });
                }
              } else {
                // If no sentiment analysis, create a single utterance
                utterances.push({
                  text: transcript.text,
                  start: 0,
                  end: audioDuration || 0,
                  sentiment: "neutral"
                });
              }
              
              // Process word-level data
              const words = [];
              if (transcript.words && Array.isArray(transcript.words)) {
                for (const word of transcript.words) {
                  words.push({
                    text: word.text,
                    start: word.start / 1000, // Convert to seconds
                    end: word.end / 1000,     // Convert to seconds
                    confidence: word.confidence || 0.9
                  });
                }
              }
              
              transcriptionResult = {
                text: transcript.text,
                provider: "assemblyai",
                confidence: 0.9, // AssemblyAI doesn't provide confidence scores directly
                wordLevelData: true,
                sentiment: transcript.sentiment,
                // Store the utterances and words in a format easier for quotation extraction
                transcription: {
                  full_text: transcript.text,
                  utterances: utterances,
                  words: words
                },
                emotion: emotions.map((item: { 
                  text: string, 
                  sentiment: string, 
                  confidence: number, 
                  start: number, 
                  end: number 
                }) => ({
                  text: item.text,
                  sentiment: item.sentiment,
                  confidence: item.confidence,
                  start: item.start,
                  end: item.end
                })),
                segments: transcript.words || [],
                entities: transcript.entities || [],
                topics: transcript.iab_categories_result?.summary || {}
              };
              console.log('AssemblyAI transcription successful!');
            }
          }
        }
      } catch (error) {
        console.error('AssemblyAI transcription error:', error);
      }
    }
    
    // If both previous services fail, use Deepgram as final fallback
    // Note: We'll implement this via direct API call since we had issues with the SDK
    if (transcriptionResult.provider === "none" && DEEPGRAM_API_KEY) {
      try {
        console.log('Attempting transcription with Deepgram API...');
        
        const deepgramResponse = await fetch('https://api.deepgram.com/v1/listen?model=nova-2&detect_language=true&punctuate=true&diarize=true', {
          method: 'POST',
          headers: {
            'Authorization': `Token ${DEEPGRAM_API_KEY}`,
            'Content-Type': 'audio/mp3'
          },
          body: audioBuffer
        });
        
        if (deepgramResponse.ok) {
          const result = await deepgramResponse.json();
          
          if (result.results && result.results.channels && result.results.channels.length > 0) {
            const transcript = result.results.channels[0].alternatives[0];
            
            // Process words for consistent format
            const words = [];
            if (transcript.words && Array.isArray(transcript.words)) {
              for (const word of transcript.words) {
                words.push({
                  text: word.word || word.text,
                  start: word.start,
                  end: word.end,
                  confidence: word.confidence || 0.8
                });
              }
            }
            
            // Create utterances from paragraphs or sentences
            const utterances = [];
            if (transcript.paragraphs && transcript.paragraphs.length > 0) {
              for (const paragraph of transcript.paragraphs) {
                utterances.push({
                  text: paragraph.text,
                  start: paragraph.start,
                  end: paragraph.end,
                  sentiment: "unknown" // Deepgram doesn't provide sentiment
                });
              }
            } else if (transcript.sentences && transcript.sentences.length > 0) {
              for (const sentence of transcript.sentences) {
                utterances.push({
                  text: sentence.text,
                  start: sentence.start,
                  end: sentence.end,
                  sentiment: "unknown"
                });
              }
            } else {
              // If no paragraphs or sentences, create a single utterance
              utterances.push({
                text: transcript.transcript,
                start: 0,
                end: audioDuration || 0,
                sentiment: "unknown"
              });
            }
            
            transcriptionResult = {
              text: transcript.transcript,
              provider: "deepgram",
              confidence: transcript.confidence,
              language: result.results.language,
              wordLevelData: true,
              // Store in standardized format for easy quotation extraction
              transcription: {
                full_text: transcript.transcript,
                utterances: utterances,
                words: words
              },
              segments: transcript.words || []
            };
            console.log('Deepgram transcription successful!');
          }
        }
      } catch (error) {
        console.error('Deepgram transcription error:', error);
      }
    }
    
    // If all transcription services fail, fall back to OpenAI Whisper if available
    if (transcriptionResult.provider === "none" && openai) {
      try {
        console.log('All primary transcription services failed, falling back to OpenAI Whisper...');
        
        // Reset the file stream position
        const newAudioFile = fs.createReadStream(audioPath);
        
        const transcriptionResponse = await openai.audio.transcriptions.create({
          file: newAudioFile,
          model: 'whisper-1',
          language: 'en',
          response_format: 'verbose_json',
          timestamp_granularities: ['word']
        });
        
        // Process word-level data if available
        const words = [];
        if (transcriptionResponse.words && Array.isArray(transcriptionResponse.words)) {
          for (const word of transcriptionResponse.words) {
            words.push({
              text: word.word,
              start: word.start,
              end: word.end,
              confidence: 0.92 // Whisper doesn't provide per-word confidence
            });
          }
        }
        
        // Process utterances from segments
        const utterances = [];
        if (transcriptionResponse.segments && transcriptionResponse.segments.length > 0) {
          for (const segment of transcriptionResponse.segments) {
            utterances.push({
              text: segment.text,
              start: segment.start,
              end: segment.end,
              sentiment: "unknown" // Whisper doesn't provide sentiment
            });
          }
        } else {
          // If no segments, create a single utterance
          utterances.push({
            text: transcriptionResponse.text,
            start: 0,
            end: audioDuration || 0,
            sentiment: "unknown"
          });
        }
        
        transcriptionResult = {
          text: transcriptionResponse.text,
          provider: "openai_whisper",
          confidence: 0.92, // Whisper is generally highly accurate
          wordLevelData: true,
          // Store in standardized format for easy quotation extraction
          transcription: {
            full_text: transcriptionResponse.text,
            utterances: utterances,
            words: words
          },
          segments: transcriptionResponse.segments || []
        };
        
        console.log('OpenAI Whisper transcription successful!');
      } catch (error) {
        console.error('OpenAI Whisper transcription error:', error);
      }
    }
    
    // Clean up temp file
    await unlinkAsync(audioPath).catch(err => console.warn('Error deleting temp audio file:', err));
    
    // If no transcription service worked
    if (transcriptionResult.provider === "none") {
      console.error('All transcription services failed');
      return {
        transcription: "Failed to transcribe audio. None of the transcription services were able to process this video.",
        transcriptionData: {
          full_text: "Failed to transcribe audio. None of the transcription services were able to process this video.",
          utterances: [],
          words: []
        },
        speechAnalysis: {
          provider: "none",
          averageConfidence: 0,
          speakingRate: 0,
          error: "All transcription services failed"
        }
      };
    }
    
    // Calculate speaking rate based on word count and duration
    const textToCount = transcriptionResult.text || transcriptionResult.transcription;
    const words = textToCount ? textToCount.split(' ').length : 0;
    const speakingRate = audioDuration > 0 ? words / audioDuration : 0;
    
    // Return standardized response format with detailed transcription data
    return {
      // Original transcription text (for backwards compatibility)
      transcription: transcriptionResult.text || transcriptionResult.transcription,
      // Complete structured transcription data for UI and quote extraction
      transcriptionData: transcriptionResult.transcription || {
        full_text: transcriptionResult.text || transcriptionResult.transcription,
        utterances: [],
        words: []
      },
      speechAnalysis: {
        provider: transcriptionResult.provider,
        averageConfidence: transcriptionResult.confidence,
        speakingRate,
        wordCount: words,
        duration: audioDuration,
        emotion: transcriptionResult.emotion,
        sentiment: transcriptionResult.sentiment,
        entities: transcriptionResult.entities,
        topics: transcriptionResult.topics,
        segments: transcriptionResult.segments
      }
    };
  } catch (error) {
    console.error('Error in audio transcription:', error);
    // Return a minimal object if transcription fails
    return {
      transcription: "Failed to transcribe audio. Please try again with clearer audio or a different video.",
      speechAnalysis: {
        provider: "error",
        averageConfidence: 0,
        speakingRate: 0,
        error: error instanceof Error ? error.message : "Unknown transcription error"
      }
    };
  }
}


// For backward compatibility
const uploadImageSchema = z.object({
  imageData: z.string(),
  sessionId: z.string(),
});

const sendMessageSchema = z.object({
  content: z.string(),
  sessionId: z.string(),
});

// Check if email service is configured
let isEmailServiceConfigured = false;
if (process.env.SENDGRID_API_KEY && process.env.SENDGRID_VERIFIED_SENDER) {
  isEmailServiceConfigured = true;
}

// Define the schema for retrieving a shared analysis
const getSharedAnalysisSchema = z.object({
  shareId: z.coerce.number(),
});

export async function registerRoutes(app: Express): Promise<Server> {
  // Text analysis endpoint
  app.post("/api/analyze/text", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "deepseek", title } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Text content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      // Choose which AI model to use
      let aiModel = selectedModel;
      if (
        (selectedModel === "deepseek" && !deepseek) ||
        (selectedModel === "openai" && !openai) ||
        (selectedModel === "anthropic" && !anthropic) ||
        (selectedModel === "perplexity" && !process.env.PERPLEXITY_API_KEY)
      ) {
        // Fallback to available model if selected one is not available
        if (openai) aiModel = "openai";
        else if (anthropic) aiModel = "anthropic";
        else if (process.env.PERPLEXITY_API_KEY) aiModel = "perplexity";
        else {
          return res.status(503).json({ 
            error: "No AI models are currently available. Please try again later." 
          });
        }
      }
      
      // Get personality insights based on text content
      let personalityInsights;
      const textAnalysisPrompt = `
You are an expert psychologist and personality analyst. Analyze the following text to provide comprehensive personality insights about the author.

CRITICAL REQUIREMENTS - FAILURE TO COMPLY WILL RESULT IN REJECTED ANALYSIS:
- NO MARKDOWN FORMATTING: Do not use # ### ** or any markdown in your response
- INCLUDE 8-12 DIRECT QUOTES from the text that reveal personality traits
- PROVIDE COMPREHENSIVE 3-4 PARAGRAPH SECTIONS for each analysis area
- EXTRACT SPECIFIC EVIDENCE from word choice, topics discussed, and communication style
- YOU MUST ANSWER ALL 50 PSYCHOLOGICAL QUESTIONS WITH EXPLICIT, SUBSTANTIVE ANSWERS
- ABSOLUTELY NO PLACEHOLDER TEXT, GENERIC STATEMENTS, OR "NOT ASSESSED" RESPONSES
- EVERY SINGLE QUESTION MUST HAVE A DETAILED, SPECIFIC ANSWER WITH EVIDENCE
- IF DATA IS LIMITED, MAKE REASONABLE PSYCHOLOGICAL INFERENCES BASED ON AVAILABLE EVIDENCE

TEXT TO ANALYZE:
${content}

You must provide detailed answers to these 50 fundamental psychological questions based on the text analysis:

I. Language & Style (10):
1. What is the dominant sentence rhythm — clipped, flowing, erratic — and what personality trait does it reveal?
2. Which adjectives recur, and what emotional bias do they show?
3. How does pronoun use ('I,' 'you,' 'we,' 'they') shift across the text, and what identity stance does that reflect?
4. What level of abstraction vs. concreteness dominates the writing?
5. Identify one passage where diction becomes suddenly elevated or deflated — what triggers it?
6. Are there unfinished or fragmentary sentences, and what might that signal psychologically?
7. How consistent is the tense? Does the writer slip between past and present, and why?
8. What metaphors or analogies recur, and what unconscious associations do they expose?
9. Is the author's tone self-assured, tentative, ironic, or performative? Cite phrasing.
10. What linguistic register (formal, colloquial, technical) dominates, and how does it align with self-image?

II. Emotional Indicators (10):
11. What emotion seems primary (anger, melancholy, pride, longing), and where is it linguistically concentrated?
12. Which emotions appear repressed or displaced — hinted at but never named?
13. Does emotional intensity rise or fall as the text progresses?
14. Identify one sentence where affect 'leaks through' despite apparent control.
15. Are there moments of sentimental overstatement or cold detachment?
16. What bodily or sensory words appear, and what do they suggest about embodiment or repression?
17. Is there ambivalence toward the subject matter? Cite a line where tone wavers.
18. Does humor appear, and if so, is it self-directed, aggressive, or defensive?
19. What words betray anxiety or guilt?
20. How is desire represented — directly, symbolically, or through avoidance?

III. Cognitive & Structural Patterns (10):
21. How logically coherent are transitions between ideas?
22. Does the writer prefer enumeration, narrative, or digression? What does that indicate about thought style?
23. What syntactic habits dominate (parallelism, repetition, parenthesis), and what mental rhythms do they mirror?
24. Are there contradictions the author fails to notice? Quote one.
25. How does the author handle uncertainty — through hedging, assertion, or silence?
26. Does the argument or story circle back on itself?
27. Are there abrupt topic shifts, and what emotional events coincide with them?
28. What elements of the text seem compulsive or ritualistic in repetition?
29. Where does the writer show real insight versus mechanical reasoning?
30. How does closure occur (resolution, withdrawal, collapse), and what does it signify psychologically?

IV. Self-Representation & Identity (10):
31. How does the writer portray the self — victim, hero, observer, analyst?
32. Is there a split between narrating voice and lived experience?
33. What form of authority or validation does the author seek (moral, intellectual, emotional)?
34. How consistent is the self-image across paragraphs?
35. Identify one phrase that reveals unconscious self-evaluation (admiration, contempt, shame).
36. Does the author reveal dependency on external approval or autonomy from it?
37. What form of vulnerability does the writer allow?
38. How does the author talk about others — with empathy, rivalry, indifference?
39. What implicit audience is being addressed?
40. Does the writer's stance shift from confession to performance? Cite turning point.

V. Symbolic & Unconscious Material (10):
41. Which images or motifs recur (light/dark, ascent/descent, enclosure, mirrors), and what do they symbolize?
42. Are there dream-like or surreal elements?
43. What oppositions structure the text (order/chaos, love/power, mind/body)?
44. What wish or fear seems to animate the text beneath the surface argument?
45. Identify one metaphor that reads like a disguised confession.
46. How does the author relate to time — nostalgic, future-oriented, frozen?
47. Does the text express conflict between intellect and emotion?
48. What shadow aspect of personality is hinted at through hostile or taboo imagery?
49. Is there evidence of projection — attributing inner states to others or to abstractions?
50. What central psychological drama (loss, control, recognition, transformation) structures the entire piece?

Respond with clean JSON (no markdown formatting anywhere):
{
  "summary": "comprehensive 2-3 paragraph overview integrating content analysis with deep personality insights and specific evidence from the text",
  "detailed_analysis": {
    "content_themes": "detailed 3-4 paragraph analysis of topics discussed, interests revealed, concerns expressed, and what these reveal about the person's priorities and character",
    "speech_analysis": {
      "key_quotes": ["quote 1", "quote 2", "quote 3", "quote 4", "quote 5", "quote 6", "quote 7", "quote 8", "quote 9", "quote 10"],
      "vocabulary_analysis": "comprehensive analysis of word choice, linguistic sophistication, communication style, and what it reveals about personality",
      "personality_revealed": "detailed insights into character traits, values, and psychological patterns revealed through their specific word choices and expressions"
    },
    "core_psychological_assessment": {
      "core_motivation": "What drives this person based on their expressed interests, priorities, and concerns",
      "confidence_level": "Assessment of their real confidence level with supporting evidence from the text",
      "self_acceptance": "Do they genuinely like themselves - evidence from self-talk and self-references",
      "intelligence_level": "How smart are they - vocabulary, reasoning, complexity of ideas",
      "creativity_assessment": "How creative are they - originality, innovative thinking, unique perspectives",
      "stress_handling": "How they handle stress or setbacks based on their responses and attitudes",
      "trustworthiness": "Are they trustworthy - consistency, honesty, reliability indicators",
      "authenticity": "Do they exaggerate or fake things - genuine vs performative elements",
      "ambition_level": "How ambitious are they - goal orientation, drive, achievement focus",
      "insecurities": "What are they insecure about - revealed through defensiveness or overcompensation",
      "social_validation": "How much do they care what others think - external vs internal validation",
      "independence": "Are they independent-minded or followers - original vs conventional thinking",
      "communication_style": "Do they dominate conversations or listen more - assertiveness patterns",
      "criticism_response": "How do they deal with criticism - defensive, receptive, or dismissive",
      "outlook": "Are they more optimistic or pessimistic - positive vs negative framing",
      "humor_sense": "Do they have a strong sense of humor - wit, playfulness, levity",
      "treatment_of_others": "How do they treat people beneath them - respect, dismissiveness, condescension",
      "consistency": "Are they consistent or contradictory - logical coherence in their positions",
      "hidden_strengths": "What hidden strengths do they have - subtle positive qualities",
      "hidden_weaknesses": "What hidden weaknesses do they have - subtle negative patterns"
    },
    "professional_insights": "comprehensive analysis of career inclinations, work style preferences, leadership qualities, professional strengths, and workplace behavior patterns",
    "growth_areas": {
      "strengths": ["strength 1 with detailed evidence", "strength 2 with evidence", "strength 3 with evidence"],
      "development_path": "detailed recommendations for personal and professional growth based on demonstrated patterns and potential areas for development"
    }
  }
}
`;

      // Get personality analysis from selected AI model
      let analysisResult;
      if (aiModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
          messages: [
            { role: "system", content: "You are an expert in personality analysis and psychological assessment." },
            { role: "user", content: textAnalysisPrompt }
          ],
          response_format: { type: "json_object" }
        });
        
        try {
          const rawContent = completion.choices[0].message.content || '{}';
          analysisResult = JSON.parse(rawContent);
          
          // Clean up any markdown formatting in the analysis content
          if (analysisResult && typeof analysisResult === 'object') {
            analysisResult = cleanMarkdownFromAnalysis(analysisResult);
          }
        } catch (parseError) {
          console.error("Error parsing OpenAI JSON response:", parseError);
          const rawContent = completion.choices[0].message.content || '';
          analysisResult = {
            summary: rawContent.substring(0, 500) + "...",
            detailed_analysis: {
              personality_core: "Unable to parse structured response. Raw analysis: " + rawContent.substring(0, 300),
              thought_patterns: "Please try refreshing or using a different AI model",
              emotional_tendencies: "",
              communication_style: ""
            }
          };
        }
      } 
      else if (aiModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514", // the newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229". 
          max_tokens: 4000,
          system: "You are an expert in personality analysis and psychological assessment. Always respond with well-structured JSON.",
          messages: [{ role: "user", content: textAnalysisPrompt }],
        });
        
        try {
          let rawContent = (response.content[0] as any).text || '{}';
          
          // Remove markdown code blocks if present
          rawContent = rawContent.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
          
          // Find the outermost JSON object more carefully
          let openBraces = 0;
          let jsonStart = -1;
          let jsonEnd = -1;
          
          for (let i = 0; i < rawContent.length; i++) {
            if (rawContent[i] === '{') {
              if (jsonStart === -1) jsonStart = i;
              openBraces++;
            } else if (rawContent[i] === '}') {
              openBraces--;
              if (openBraces === 0 && jsonStart !== -1) {
                jsonEnd = i;
                break;
              }
            }
          }
          
          if (jsonStart !== -1 && jsonEnd !== -1) {
            rawContent = rawContent.substring(jsonStart, jsonEnd + 1);
          }
          
          analysisResult = JSON.parse(rawContent);
          
          // Clean up any markdown formatting in the analysis content
          if (analysisResult && typeof analysisResult === 'object') {
            analysisResult = cleanMarkdownFromAnalysis(analysisResult);
          }
        } catch (parseError) {
          console.error("Error parsing Anthropic JSON response:", parseError);
          const rawContent = (response.content[0] as any).text || '';
          analysisResult = {
            summary: rawContent.substring(0, 500) + "...",
            detailed_analysis: {
              personality_core: "Unable to parse structured response. Raw analysis: " + rawContent.substring(0, 300),
              thought_patterns: "Please try refreshing or using a different AI model",
              emotional_tendencies: "",
              communication_style: ""
            }
          };
        }
      }
      else if (aiModel === "perplexity") {
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: textAnalysisPrompt
        });
        
        try {
          let cleanText = response.text.replace(/```json\s*/g, '').replace(/```\s*/g, '').trim();
          
          // Find the outermost JSON object more carefully
          let openBraces = 0;
          let jsonStart = -1;
          let jsonEnd = -1;
          
          for (let i = 0; i < cleanText.length; i++) {
            if (cleanText[i] === '{') {
              if (jsonStart === -1) jsonStart = i;
              openBraces++;
            } else if (cleanText[i] === '}') {
              openBraces--;
              if (openBraces === 0 && jsonStart !== -1) {
                jsonEnd = i;
                break;
              }
            }
          }
          
          if (jsonStart !== -1 && jsonEnd !== -1) {
            cleanText = cleanText.substring(jsonStart, jsonEnd + 1);
          }
          
          analysisResult = JSON.parse(cleanText);
          
          // Clean up any markdown formatting in the analysis content
          if (analysisResult && typeof analysisResult === 'object') {
            analysisResult = cleanMarkdownFromAnalysis(analysisResult);
          }
        } catch (e) {
          console.error("Error parsing Perplexity response:", e);
          // Fallback structure if parsing fails
          analysisResult = {
            summary: response.text.substring(0, 200) + "...",
            detailed_analysis: {
              personality_core: "Error parsing structured response from Perplexity",
              thought_patterns: "Please try again with a different AI model"
            }
          };
        }
      }
      
      // VALIDATE: Ensure all 20 core psychological questions are answered
      validateCoreAssessment(analysisResult, "Text Analysis Subject");
      
      // Create personality insights in expected format
      personalityInsights = {
        peopleCount: 1,
        individualProfiles: [analysisResult]
      };
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: "", // Text analysis doesn't require a media URL
        mediaType: "text",
        personalityInsights,
        title: title || "Text Analysis"
      });
      
      // Format message for response with comprehensive psychological assessment
      const coreAssessment = analysisResult.detailed_analysis?.core_psychological_assessment;
      const formattedContent = `
Personality Analysis Based on Text

${analysisResult.summary || "Analysis summary not available"}

Core Psychological Assessment

What drives this person: ${coreAssessment?.core_motivation || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Confidence level: ${coreAssessment?.confidence_level || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Self-acceptance: ${coreAssessment?.self_acceptance || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Intelligence level: ${coreAssessment?.intelligence_level || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Creativity: ${coreAssessment?.creativity_assessment || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Stress handling: ${coreAssessment?.stress_handling || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Trustworthiness: ${coreAssessment?.trustworthiness || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Authenticity: ${coreAssessment?.authenticity || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Ambition level: ${coreAssessment?.ambition_level || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Insecurities: ${coreAssessment?.insecurities || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Social validation needs: ${coreAssessment?.social_validation || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Independence: ${coreAssessment?.independence || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Communication style: ${coreAssessment?.communication_style || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Response to criticism: ${coreAssessment?.criticism_response || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Outlook: ${coreAssessment?.outlook || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Sense of humor: ${coreAssessment?.humor_sense || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Treatment of others: ${coreAssessment?.treatment_of_others || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Consistency: ${coreAssessment?.consistency || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Hidden strengths: ${coreAssessment?.hidden_strengths || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Hidden weaknesses: ${coreAssessment?.hidden_weaknesses || "[ERROR: AI model failed to provide answer - please regenerate analysis]"}

Speech Analysis & Quotes
${analysisResult.detailed_analysis?.speech_analysis ? 
  `Key Quotes: ${(analysisResult.detailed_analysis.speech_analysis.key_quotes || []).join(' | ')}

Vocabulary Analysis: ${analysisResult.detailed_analysis.speech_analysis.vocabulary_analysis || "Not available"}

Personality Revealed: ${analysisResult.detailed_analysis.speech_analysis.personality_revealed || "Not available"}` 
  : "Speech analysis not available"}

Content Themes
${analysisResult.detailed_analysis?.content_themes || "Content analysis not available"}

Professional Insights
${analysisResult.detailed_analysis?.professional_insights || "Professional insights not available"}

Growth Areas
${analysisResult.detailed_analysis?.growth_areas ? 
  `Strengths: ${(analysisResult.detailed_analysis.growth_areas.strengths || []).join(', ')}

Development Path: ${analysisResult.detailed_analysis.growth_areas.development_path || "Not available"}`
  : "Growth analysis not available"}

You can ask follow-up questions about this analysis.
`;
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: formattedContent
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Text analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze text" });
      }
    }
  });
  
  // Document analysis endpoint
  app.post("/api/analyze/document", async (req, res) => {
    try {
      const { fileData, fileName, fileType, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!fileData || typeof fileData !== 'string') {
        return res.status(400).json({ error: "Document data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      // Extract base64 content from data URL
      const base64Data = fileData.split(',')[1];
      if (!base64Data) {
        return res.status(400).json({ error: "Invalid document data format" });
      }
      
      // Save the document to a temporary file
      const fileBuffer = Buffer.from(base64Data, 'base64');
      const tempDocPath = path.join(tempDir, `doc_${Date.now()}_${fileName}`);
      await writeFileAsync(tempDocPath, fileBuffer);
      
      // Choose which AI model to use
      let aiModel = selectedModel;
      if (
        (selectedModel === "openai" && !openai) ||
        (selectedModel === "anthropic" && !anthropic) ||
        (selectedModel === "perplexity" && !process.env.PERPLEXITY_API_KEY)
      ) {
        // Fallback to available model if selected one is not available
        if (openai) aiModel = "openai";
        else if (anthropic) aiModel = "anthropic";
        else if (process.env.PERPLEXITY_API_KEY) aiModel = "perplexity";
        else {
          return res.status(503).json({ 
            error: "No AI models are currently available. Please try again later." 
          });
        }
      }
      
      // Extract text from document and analyze it
      // Note: In a real implementation, use proper document parsing libraries
      // like pdf.js, docx, etc. For simplicity, we're using a placeholder.
      const documentAnalysisPrompt = `
You are an expert psychologist and personality analyst. Analyze the uploaded document: ${fileName} (${fileType}).

CRITICAL REQUIREMENTS:
- NO MARKDOWN FORMATTING: Do not use # ### ** or any markdown in your response
- PROVIDE COMPREHENSIVE 3-4 PARAGRAPH SECTIONS for each analysis area
- EXTRACT SPECIFIC EVIDENCE from writing style, content, and communication patterns
- ANSWER ALL 50 PSYCHOLOGICAL QUESTIONS with specific evidence

You must provide detailed answers to these 50 fundamental psychological questions based on the document analysis:

I. Language & Style (10):
1. What is the dominant sentence rhythm — clipped, flowing, erratic — and what personality trait does it reveal?
2. Which adjectives recur, and what emotional bias do they show?
3. How does pronoun use ('I,' 'you,' 'we,' 'they') shift across the text, and what identity stance does that reflect?
4. What level of abstraction vs. concreteness dominates the writing?
5. Identify one passage where diction becomes suddenly elevated or deflated — what triggers it?
6. Are there unfinished or fragmentary sentences, and what might that signal psychologically?
7. How consistent is the tense? Does the writer slip between past and present, and why?
8. What metaphors or analogies recur, and what unconscious associations do they expose?
9. Is the author's tone self-assured, tentative, ironic, or performative? Cite phrasing.
10. What linguistic register (formal, colloquial, technical) dominates, and how does it align with self-image?

II. Emotional Indicators (10):
11. What emotion seems primary (anger, melancholy, pride, longing), and where is it linguistically concentrated?
12. Which emotions appear repressed or displaced — hinted at but never named?
13. Does emotional intensity rise or fall as the text progresses?
14. Identify one sentence where affect 'leaks through' despite apparent control.
15. Are there moments of sentimental overstatement or cold detachment?
16. What bodily or sensory words appear, and what do they suggest about embodiment or repression?
17. Is there ambivalence toward the subject matter? Cite a line where tone wavers.
18. Does humor appear, and if so, is it self-directed, aggressive, or defensive?
19. What words betray anxiety or guilt?
20. How is desire represented — directly, symbolically, or through avoidance?

III. Cognitive & Structural Patterns (10):
21. How logically coherent are transitions between ideas?
22. Does the writer prefer enumeration, narrative, or digression? What does that indicate about thought style?
23. What syntactic habits dominate (parallelism, repetition, parenthesis), and what mental rhythms do they mirror?
24. Are there contradictions the author fails to notice? Quote one.
25. How does the author handle uncertainty — through hedging, assertion, or silence?
26. Does the argument or story circle back on itself?
27. Are there abrupt topic shifts, and what emotional events coincide with them?
28. What elements of the text seem compulsive or ritualistic in repetition?
29. Where does the writer show real insight versus mechanical reasoning?
30. How does closure occur (resolution, withdrawal, collapse), and what does it signify psychologically?

IV. Self-Representation & Identity (10):
31. How does the writer portray the self — victim, hero, observer, analyst?
32. Is there a split between narrating voice and lived experience?
33. What form of authority or validation does the author seek (moral, intellectual, emotional)?
34. How consistent is the self-image across paragraphs?
35. Identify one phrase that reveals unconscious self-evaluation (admiration, contempt, shame).
36. Does the author reveal dependency on external approval or autonomy from it?
37. What form of vulnerability does the writer allow?
38. How does the author talk about others — with empathy, rivalry, indifference?
39. What implicit audience is being addressed?
40. Does the writer's stance shift from confession to performance? Cite turning point.

V. Symbolic & Unconscious Material (10):
41. Which images or motifs recur (light/dark, ascent/descent, enclosure, mirrors), and what do they symbolize?
42. Are there dream-like or surreal elements?
43. What oppositions structure the text (order/chaos, love/power, mind/body)?
44. What wish or fear seems to animate the text beneath the surface argument?
45. Identify one metaphor that reads like a disguised confession.
46. How does the author relate to time — nostalgic, future-oriented, frozen?
47. Does the text express conflict between intellect and emotion?
48. What shadow aspect of personality is hinted at through hostile or taboo imagery?
49. Is there evidence of projection — attributing inner states to others or to abstractions?
50. What central psychological drama (loss, control, recognition, transformation) structures the entire piece?

Respond with clean JSON (no markdown formatting anywhere):
{
  "summary": "comprehensive 2-3 paragraph overview integrating document analysis with deep personality insights",
  "detailed_analysis": {
    "document_overview": "detailed analysis of document type, structure, content themes, and what they reveal about the author",
    "core_psychological_assessment": {
      "core_motivation": "What drives this person based on their expressed interests, priorities, and concerns",
      "confidence_level": "Assessment of their real confidence level with supporting evidence from the document",
      "self_acceptance": "Do they genuinely like themselves - evidence from self-talk and self-references",
      "intelligence_level": "How smart are they - vocabulary, reasoning, complexity of ideas",
      "creativity_assessment": "How creative are they - originality, innovative thinking, unique perspectives",
      "stress_handling": "How they handle stress or setbacks based on their responses and attitudes",
      "trustworthiness": "Are they trustworthy - consistency, honesty, reliability indicators",
      "authenticity": "Do they exaggerate or fake things - genuine vs performative elements",
      "ambition_level": "How ambitious are they - goal orientation, drive, achievement focus",
      "insecurities": "What are they insecure about - revealed through defensiveness or overcompensation",
      "social_validation": "How much do they care what others think - external vs internal validation",
      "independence": "Are they independent-minded or followers - original vs conventional thinking",
      "communication_style": "Do they dominate conversations or listen more - assertiveness patterns",
      "criticism_response": "How do they deal with criticism - defensive, receptive, or dismissive",
      "outlook": "Are they more optimistic or pessimistic - positive vs negative framing",
      "humor_sense": "Do they have a strong sense of humor - wit, playfulness, levity",
      "treatment_of_others": "How do they treat people beneath them - respect, dismissiveness, condescension",
      "consistency": "Are they consistent or contradictory - logical coherence in their positions",
      "hidden_strengths": "What hidden strengths do they have - subtle positive qualities",
      "hidden_weaknesses": "What hidden weaknesses do they have - subtle negative patterns"
    },
    "writing_style": "comprehensive analysis of linguistic patterns, vocabulary, structure, and what it reveals about personality",
    "professional_insights": "career inclinations, work style preferences, leadership qualities based on document content"
  }
}
`;

      // Get document analysis from selected AI model
      let analysisResult;
      if (aiModel === "openai" && openai) {
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024. do not change this unless explicitly requested by the user
          messages: [
            { role: "system", content: "You are an expert in document analysis and personality assessment." },
            { role: "user", content: documentAnalysisPrompt }
          ],
          response_format: { type: "json_object" }
        });
        
        analysisResult = JSON.parse(completion.choices[0].message.content || '{}');
      } 
      else if (aiModel === "anthropic" && anthropic) {
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514", // the newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229". 
          max_tokens: 4000,
          system: "You are an expert in document analysis and psychological assessment. Always respond with well-structured JSON.",
          messages: [{ role: "user", content: documentAnalysisPrompt }],
        });
        
        analysisResult = JSON.parse((response.content[0] as any).text || '{}');
      }
      else if (aiModel === "perplexity") {
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: documentAnalysisPrompt
        });
        
        try {
          analysisResult = JSON.parse(response.text);
        } catch (e) {
          console.error("Error parsing Perplexity response:", e);
          // Fallback structure if parsing fails
          analysisResult = {
            summary: response.text.substring(0, 200) + "...",
            detailed_analysis: {
              document_overview: "Error parsing structured response from Perplexity",
              main_themes: "Please try again with a different AI model"
            }
          };
        }
      }
      
      // Create personality insights in expected format
      const personalityInsights = {
        peopleCount: 1,
        individualProfiles: [{
          summary: analysisResult.summary,
          detailed_analysis: {
            personality_core: analysisResult.detailed_analysis.author_personality,
            thought_patterns: analysisResult.detailed_analysis.main_themes,
            emotional_tendencies: analysisResult.detailed_analysis.emotional_tone,
            communication_style: analysisResult.detailed_analysis.writing_style
          }
        }]
      };
      
      // Clean up temporary file
      try {
        await unlinkAsync(tempDocPath);
      } catch (e) {
        console.warn("Error removing temporary document file:", e);
      }
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: "", // Document analysis uses file content, not URL
        mediaType: "document",
        personalityInsights,
        title: title || fileName
      });
      
      // Format message for response
      const formattedContent = `
# Document Analysis: ${fileName}

${analysisResult.summary}

## Document Overview
${analysisResult.detailed_analysis.document_overview}

## Main Themes
${analysisResult.detailed_analysis.main_themes}

## Emotional Tone
${analysisResult.detailed_analysis.emotional_tone}

## Writing Style
${analysisResult.detailed_analysis.writing_style}

## Author Personality Assessment
${analysisResult.detailed_analysis.author_personality}

You can ask follow-up questions about this analysis.
`;
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: formattedContent
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Document analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze document" });
      }
    }
  });
  // Text analysis endpoint
  app.post("/api/analyze/text", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Text content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing text analysis with model: ${selectedModel}`);
      
      // Use the 50-question text analysis framework
      const questions = TEXT_ANALYSIS_QUESTIONS;
      const questionCount = questions.length;
      
      // Get personality insights based on text content
      const textAnalysisPrompt = `
Please analyze the following text to provide comprehensive personality insights about the author.

YOU MUST ANSWER ALL ${questionCount} QUESTIONS BELOW:

TEXT:
${content}

Based on this text, answer these ${questionCount} psychological questions about the author. For each question, provide specific evidence from the text including 8-12 direct quotations that support your assessment. Do not use markdown formatting - use plain text only.

PSYCHOLOGICAL QUESTIONS TO ANSWER:
${questions.map((q, i) => `${i + 1}. ${q}`).join('\n')}

ANALYSIS REQUIREMENTS:
- Extract 8-12 direct quotations from the text as supporting evidence
- Analyze writing style, tone, word choice, and content themes
- Provide detailed psychological assessment for each question
- Include specific textual evidence for every psychological conclusion
- Focus on observable patterns in language and communication style
- Address cognitive processing style, emotional expression, and personality traits

FORMAT: Answer each question thoroughly with specific evidence. Include direct quotes in quotation marks. Use plain text formatting without markdown headers or special characters.
`;

      // Get personality analysis from selected AI model
      let analysisText: string;
      
      if (selectedModel === "openai" && openai) {
        console.log('Using OpenAI for text analysis');
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024
          messages: [
            { role: "system", content: "You are an expert in personality analysis and psychological assessment." },
            { role: "user", content: textAnalysisPrompt }
          ]
        });
        
        analysisText = completion.choices[0].message.content || "";
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        console.log('Using Anthropic for text analysis');
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219", // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
          max_tokens: 4000,
          system: "You are an expert in personality analysis and psychological assessment.",
          messages: [{ role: "user", content: textAnalysisPrompt }],
        });
        
        analysisText = response.content[0].text;
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        console.log('Using Perplexity for text analysis');
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: textAnalysisPrompt
        });
        
        analysisText = response.text;
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Create an analysis with a dummy mediaUrl since the schema requires it but we don't have
      // media for text analysis
      const dummyMediaUrl = `text:${Date.now()}`;
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: dummyMediaUrl,
        mediaType: "text",
        personalityInsights: { analysis: analysisText },
        title: title || "Text Analysis"
      });
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: analysisText
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Text analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze text" });
      }
    }
  });
  
  // Document analysis endpoint
  app.post("/api/analyze/document", async (req, res) => {
    try {
      const { fileData, fileName, fileType, sessionId, selectedModel = "openai", title } = req.body;
      
      if (!fileData || typeof fileData !== 'string') {
        return res.status(400).json({ error: "Document data is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing document analysis with model: ${selectedModel}, file: ${fileName}`);
      
      // Use the 50-question text analysis framework (documents are analyzed as text)
      const questions = TEXT_ANALYSIS_QUESTIONS;
      const questionCount = questions.length;
      
      // Extract base64 content from data URL
      const base64Data = fileData.split(',')[1];
      if (!base64Data) {
        return res.status(400).json({ error: "Invalid document data format" });
      }
      
      // Save the document to a temporary file
      const fileBuffer = Buffer.from(base64Data, 'base64');
      const tempDocPath = path.join(tempDir, `doc_${Date.now()}_${fileName}`);
      await writeFileAsync(tempDocPath, fileBuffer);
      
      // Document analysis prompt with depth-based questions
      const documentAnalysisPrompt = `
Analyze this document: ${fileName} (${fileType}) to provide comprehensive psychological insights about the author.

Answer these ${questionCount} psychological questions about the author based on their writing. For each question, provide specific evidence from the document including 8-12 direct quotations that support your assessment. Do not use markdown formatting - use plain text only.

PSYCHOLOGICAL QUESTIONS TO ANSWER:
${questions.map((q, i) => `${i + 1}. ${q}`).join('\n')}

ANALYSIS REQUIREMENTS:
- Extract 8-12 direct quotations from the document as supporting evidence
- Analyze writing style, tone, word choice, and content themes
- Provide detailed psychological assessment for each question
- Include specific textual evidence for every psychological conclusion
- Focus on observable patterns in language and communication style
- Address cognitive processing style, emotional expression, and personality traits

FORMAT: Answer each question thoroughly with specific evidence. Include direct quotes in quotation marks. Use plain text formatting without markdown headers or special characters.
`;

      // Get document analysis from selected AI model
      let analysisText: string;
      
      if (selectedModel === "openai" && openai) {
        console.log('Using OpenAI for document analysis');
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024
          messages: [
            { role: "system", content: "You are an expert in document analysis and personality assessment." },
            { role: "user", content: documentAnalysisPrompt }
          ]
        });
        
        analysisText = completion.choices[0].message.content || "";
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        console.log('Using Anthropic for document analysis');
        const response = await anthropic.messages.create({
          model: "claude-3-7-sonnet-20250219", // the newest Anthropic model is "claude-3-7-sonnet-20250219" which was released February 24, 2025
          max_tokens: 4000,
          system: "You are an expert in document analysis and psychological assessment.",
          messages: [{ role: "user", content: documentAnalysisPrompt }],
        });
        
        analysisText = response.content[0].text;
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        console.log('Using Perplexity for document analysis');
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: documentAnalysisPrompt
        });
        
        analysisText = response.text;
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Clean up temporary file
      try {
        await unlinkAsync(tempDocPath);
      } catch (e) {
        console.warn("Error removing temporary document file:", e);
      }
      
      // Create an analysis with a dummy mediaUrl since the schema requires it but we don't have media for document analysis
      const dummyMediaUrl = `document:${Date.now()}`;
      
      // Create analysis record in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: dummyMediaUrl,
        mediaType: "document",
        personalityInsights: { analysis: analysisText },
        documentType: fileType === "pdf" ? "pdf" : "docx",
        title: title || fileName
      });
      
      // Create initial message
      const initialMessage = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        role: "assistant",
        content: analysisText
      });
      
      // Return data to client
      res.json({
        analysisId: analysis.id,
        messages: [initialMessage],
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Document analysis error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to analyze document" });
      }
    }
  });
  
  // Chat endpoint to continue conversation with AI
  app.post("/api/chat", async (req, res) => {
    try {
      const { content, sessionId, selectedModel = "openai" } = req.body;
      
      if (!content || typeof content !== 'string') {
        return res.status(400).json({ error: "Message content is required" });
      }
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      console.log(`Processing chat with model: ${selectedModel}, sessionId: ${sessionId}`);
      
      // Get existing messages for this session
      const existingMessages = await storage.getMessagesBySessionId(sessionId);
      const analysisId = existingMessages.length > 0 ? existingMessages[0].analysisId : null;
      
      // Create user message
      const userMessage = await storage.createMessage({
        sessionId,
        analysisId,
        role: "user",
        content
      });
      
      // Get analysis if available
      let analysisContext = "";
      if (analysisId) {
        const analysis = await storage.getAnalysisById(analysisId);
        if (analysis && analysis.personalityInsights) {
          // Add the analysis context for better AI responses
          analysisContext = "This conversation is about a personality analysis. Here's the context: " + 
            JSON.stringify(analysis.personalityInsights);
        }
      }
      
      // Format the conversation history for the AI
      const conversationHistory = existingMessages.map(msg => ({
        role: msg.role,
        content: msg.content
      }));
      
      // Add the new user message
      conversationHistory.push({
        role: "user",
        content
      });
      
      // Get AI response based on selected model
      let aiResponseText: string;
      
      if (selectedModel === "openai" && openai) {
        console.log('Using OpenAI for chat');
        const systemPrompt = analysisContext ? 
          `You are an AI assistant specialized in personality analysis. 

CRITICAL FORMATTING RULE: Do not use ANY markdown formatting in your response. Do not use hashtags (#), asterisks (*), underscores (_), or any other markdown symbols. Use plain text only with clear paragraph breaks and simple text formatting.

${analysisContext}

Provide detailed, comprehensive psychological insights based on the analysis. Always respond in plain text format without any markdown symbols.` :
          "You are an AI assistant specialized in personality analysis. Be helpful, informative, and engaging. CRITICAL: Do not use ANY markdown formatting (no #, *, _, etc.) - use plain text only.";
        
        const completion = await openai.chat.completions.create({
          model: "gpt-4o", // the newest OpenAI model is "gpt-4o" which was released May 13, 2024
          messages: [
            { 
              role: "system", 
              content: systemPrompt
            },
            ...conversationHistory.map(msg => ({
              role: msg.role as any,
              content: msg.content
            }))
          ]
        });
        
        aiResponseText = completion.choices[0].message.content || "";
      } 
      else if (selectedModel === "anthropic" && anthropic) {
        console.log('Using Anthropic for chat');
        const systemPrompt = analysisContext ? 
          `You are an AI assistant specialized in personality analysis. 

CRITICAL FORMATTING RULE: Do not use ANY markdown formatting in your response. Do not use hashtags (#), asterisks (*), underscores (_), or any other markdown symbols. Use plain text only with clear paragraph breaks and simple text formatting.

${analysisContext}

Provide detailed, comprehensive psychological insights based on the analysis. Always respond in plain text format without any markdown symbols.` :
          "You are an AI assistant specialized in personality analysis. Be helpful, informative, and engaging. CRITICAL: Do not use ANY markdown formatting (no #, *, _, etc.) - use plain text only.";
          
        // Format conversation history for Claude
        const messages = conversationHistory.map(msg => ({
          role: msg.role as any, 
          content: msg.content
        }));
        
        const response = await anthropic.messages.create({
          model: "claude-sonnet-4-20250514", // the newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229". 
          max_tokens: 4000,
          system: systemPrompt,
          messages
        });
        
        aiResponseText = (response.content[0] as any).text || "";
      }
      else if (selectedModel === "perplexity" && process.env.PERPLEXITY_API_KEY) {
        console.log('Using Perplexity for chat');
        // Format conversation for Perplexity
        // We need to format the entire conversation as a single prompt
        let formattedConversation = "You are an AI assistant specialized in personality analysis. CRITICAL: Do not use ANY markdown formatting (no #, *, _, etc.) - use plain text only. ";
        if (analysisContext) {
          formattedConversation += analysisContext + "\n\n";
        }
        
        formattedConversation += "Here's the conversation so far:\n\n";
        
        for (const message of conversationHistory) {
          formattedConversation += `${message.role === 'user' ? 'User' : 'Assistant'}: ${message.content}\n\n`;
        }
        
        formattedConversation += "Please provide your next response as the assistant:";
        
        const response = await perplexity.query({
          model: "llama-3.1-sonar-small-128k-online",
          query: formattedConversation
        });
        
        aiResponseText = response.text;
      }
      else {
        return res.status(503).json({ 
          error: "Selected AI model is not available. Please try again with a different model." 
        });
      }
      
      // Create AI response message
      const aiMessage = await storage.createMessage({
        sessionId,
        analysisId,
        role: "assistant",
        content: aiResponseText
      });
      
      // Return both the user message and AI response
      res.json({
        messages: [userMessage, aiMessage],
        success: true
      });
    } catch (error) {
      console.error("Chat error:", error);
      if (error instanceof Error) {
        res.status(500).json({ error: error.message });
      } else {
        res.status(500).json({ error: "Failed to process chat message" });
      }
    }
  });

  app.post("/api/analyze", async (req, res) => {
    try {
      // Use the new schema that supports both image and video with optional maxPeople
      const { mediaData, mediaType, sessionId, maxPeople = 5, selectedModel = "deepseek", videoSegmentStart = 0, videoSegmentDuration = 3 } = uploadMediaSchema.parse(req.body);

      // Extract base64 data
      const base64Data = mediaData.replace(/^data:(image|video)\/\w+;base64,/, "");
      const mediaBuffer = Buffer.from(base64Data, 'base64');

      let faceAnalysis: any = [];
      let videoAnalysis: any = null;
      let audioTranscription: any = null;
      
      // Process based on media type
      if (mediaType === "image") {
        // For images, use comprehensive multi-service face analysis
        console.log(`Analyzing image for up to ${maxPeople} people using ALL available services...`);
        faceAnalysis = await comprehensiveMultiServiceFaceAnalysis(mediaBuffer, maxPeople);
        console.log(`Detected ${Array.isArray(faceAnalysis) ? faceAnalysis.length : 1} people in the image`);
      } else {
        // For videos, we use the new 3-second segment approach
        try {
          console.log(`Video size: ${mediaBuffer.length / 1024 / 1024} MB`);
          
          // Save video to temp file
          const randomId = Math.random().toString(36).substring(2, 15);
          const videoPath = path.join(tempDir, `${randomId}.mp4`);
          
          // Write the video file temporarily
          await writeFileAsync(videoPath, mediaBuffer);
          
          // Get video duration using ffprobe
          const videoDuration = await getVideoDuration(videoPath);
          console.log(`Video duration: ${videoDuration} seconds`);
          
          // Extract the specific 3-second segment requested
          const segmentPath = path.join(tempDir, `${randomId}_segment.mp4`);
          const actualDuration = Math.min(videoSegmentDuration, videoDuration - videoSegmentStart);
          
          if (actualDuration <= 0) {
            throw new Error(`Invalid segment: starts at ${videoSegmentStart}s but video is only ${videoDuration}s long`);
          }
          
          console.log(`Extracting ${actualDuration}s segment starting at ${videoSegmentStart}s...`);
          await extractVideoSegment(videoPath, videoSegmentStart, actualDuration, segmentPath);
          
          // Process the segment instead of the full video
          const segmentBuffer = await fs.promises.readFile(segmentPath);
          
          // Extract a frame from the segment for facial analysis
          const frameExtractionPath = path.join(tempDir, `${randomId}_frame.jpg`);
          
          // Use ffmpeg to extract a frame from the segment
          await new Promise<void>((resolve, reject) => {
            ffmpeg(segmentPath)
              .screenshots({
                timestamps: ['50%'], // Take a screenshot at 50% of the segment
                filename: `${randomId}_frame.jpg`,
                folder: tempDir,
                size: '640x480'
              })
              .on('end', () => resolve())
              .on('error', (err: Error) => reject(err));
          });
          
          // Extract a frame for face analysis
          const frameBuffer = await fs.promises.readFile(frameExtractionPath);
          
          // Now run comprehensive multi-service face analysis on the extracted frame
          faceAnalysis = await comprehensiveMultiServiceFaceAnalysis(frameBuffer, maxPeople);
          console.log(`Detected ${Array.isArray(faceAnalysis) ? faceAnalysis.length : 1} people in the video frame`);
          
          // Process the segment for comprehensive analysis
          console.log(`Processing video segment: ${videoSegmentStart}s to ${videoSegmentStart + actualDuration}s`);
          
          // Try to get Azure Video Indexer analysis if available (on the segment)
          let azureVideoInsights = null;
          
          if (AZURE_VIDEO_INDEXER_KEY && AZURE_VIDEO_INDEXER_LOCATION && AZURE_VIDEO_INDEXER_ACCOUNT_ID) {
            try {
              console.log('Attempting deep video analysis with Azure Video Indexer...');
              azureVideoInsights = await analyzeVideoWithAzureIndexer(segmentBuffer);
              
              if (azureVideoInsights) {
                console.log('Azure Video Indexer analysis successful!');
              }
            } catch (error) {
              console.warn('Azure Video Indexer analysis failed:', error);
              // Continue with basic analysis if Azure Video Indexer fails
            }
          }
          
          // Create a comprehensive video analysis for the segment
          videoAnalysis = {
            provider: azureVideoInsights ? "azure_video_indexer" : "basic",
            segmentStart: videoSegmentStart,
            segmentDuration: actualDuration,
            totalVideoDuration: videoDuration,
            segmentData: {
              timestamp: videoSegmentStart,
              duration: actualDuration,
              faceAnalysis: faceAnalysis
            },
            
            // Include Azure insights if available
            ...(azureVideoInsights && { azureInsights: azureVideoInsights })
          };
          
          // Get audio transcription from the segment
          console.log('Starting audio transcription with Whisper API...');
          audioTranscription = await extractAudioTranscription(segmentPath);
          console.log(`Audio transcription complete. Text length: ${audioTranscription.transcription.length} characters`);
          
          // Clean up temp files
          try {
            // Remove the main video file, segment, and frame
            await unlinkAsync(videoPath);
            await unlinkAsync(segmentPath);
            await unlinkAsync(frameExtractionPath);
          } catch (e) {
            console.warn("Error cleaning up temp files:", e);
          }
        } catch (error) {
          console.error("Error processing video:", error);
          throw new Error("Failed to process video. Please try a smaller video file or an image.");
        }
      }

      // Get comprehensive personality insights with enhanced cognitive profiling
      const personalityInsights = await getEnhancedPersonalityInsights(
        faceAnalysis, 
        videoAnalysis, 
        audioTranscription,
        selectedModel
      );

      // Determine how many people were detected
      const peopleCount = personalityInsights.peopleCount || 1;

      // Create analysis in storage
      const analysis = await storage.createAnalysis({
        sessionId,
        mediaUrl: mediaData,
        mediaType,
        faceAnalysis,
        videoAnalysis: videoAnalysis || undefined,
        audioTranscription: audioTranscription || undefined,
        personalityInsights,
      });

      // Format initial message content for the chat
      let formattedContent = "";
      
      if (personalityInsights.individualProfiles?.length > 1) {
        // Multi-person message format with improved visual structure
        const peopleCount = personalityInsights.individualProfiles.length;
        formattedContent = `AI-Powered Psychological Profile Report\n`;
        formattedContent += `Subjects Detected: ${peopleCount} Individuals\n`;
        formattedContent += `Mode: Group Analysis\n\n`;
        
        // Add each individual profile first
        personalityInsights.individualProfiles.forEach((profile, index) => {
          const gender = profile.personLabel?.includes('Male') ? 'Male' : 
                         profile.personLabel?.includes('Female') ? 'Female' : '';
          const ageMatch = profile.personLabel?.match(/~(\d+)-(\d+)/);
          const ageRange = ageMatch ? `~${ageMatch[1]}–${ageMatch[2]} years` : '';
          const genderAge = [gender, ageRange].filter(Boolean).join(', ');
          
          formattedContent += `Subject ${index + 1}${genderAge ? ` (${genderAge})` : ''}\n`;
          formattedContent += `${'─'.repeat(40)}\n\n`;
          
          const detailedAnalysis = profile.detailed_analysis || {};
          
          formattedContent += `Summary:\n${profile.summary || 'No summary available'}\n\n`;
          
          // Display 20 Core Psychological Questions for each person
          const coreAssessment = detailedAnalysis.core_psychological_assessment || {};
          
          formattedContent += `Core Psychological Assessment:\n\n`;
          formattedContent += `What drives this person: ${coreAssessment.core_motivation || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Confidence level: ${coreAssessment.confidence_level || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Self-acceptance: ${coreAssessment.self_acceptance || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Intelligence level: ${coreAssessment.intelligence_level || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Creativity: ${coreAssessment.creativity_assessment || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Stress handling: ${coreAssessment.stress_handling || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Trustworthiness: ${coreAssessment.trustworthness || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Authenticity: ${coreAssessment.authenticity || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Ambition level: ${coreAssessment.ambition_level || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Insecurities: ${coreAssessment.insecurities || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Social validation needs: ${coreAssessment.social_validation || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Independence: ${coreAssessment.independence || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Communication style: ${coreAssessment.communication_style || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Response to criticism: ${coreAssessment.criticism_response || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Outlook: ${coreAssessment.outlook || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Sense of humor: ${coreAssessment.humor_sense || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Treatment of others: ${coreAssessment.treatment_of_others || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Consistency: ${coreAssessment.consistency || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Hidden strengths: ${coreAssessment.hidden_strengths || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n`;
          formattedContent += `Hidden weaknesses: ${coreAssessment.hidden_weaknesses || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
          
          if (detailedAnalysis.professional_insights) {
            formattedContent += `Professional Insights:\n${detailedAnalysis.professional_insights}\n\n`;
          }
          
          if (detailedAnalysis.relationships) {
            formattedContent += `Relationships:\n`;
            const relationshipParts = [];
            
            if (detailedAnalysis.relationships.current_status && 
                detailedAnalysis.relationships.current_status !== 'Not available') {
              relationshipParts.push(detailedAnalysis.relationships.current_status);
            }
            
            if (detailedAnalysis.relationships.parental_status && 
                detailedAnalysis.relationships.parental_status !== 'Not available') {
              relationshipParts.push(detailedAnalysis.relationships.parental_status);
            }
            
            if (detailedAnalysis.relationships.ideal_partner && 
                detailedAnalysis.relationships.ideal_partner !== 'Not available') {
              relationshipParts.push(`Ideal match: ${detailedAnalysis.relationships.ideal_partner}`);
            }
            
            formattedContent += relationshipParts.length > 0 
              ? relationshipParts.join(' ') 
              : 'No relationship data available';
            
            formattedContent += `\n\n`;
          }
          
          if (detailedAnalysis.growth_areas) {
            formattedContent += `📈 Growth Areas:\n`;
            
            if (Array.isArray(detailedAnalysis.growth_areas.strengths) && 
                detailedAnalysis.growth_areas.strengths.length > 0) {
              formattedContent += `Strengths:\n${detailedAnalysis.growth_areas.strengths.map((s: string) => `• ${s}`).join('\n')}\n\n`;
            }
            
            if (Array.isArray(detailedAnalysis.growth_areas.challenges) && 
                detailedAnalysis.growth_areas.challenges.length > 0) {
              formattedContent += `Challenges:\n${detailedAnalysis.growth_areas.challenges.map((c: string) => `• ${c}`).join('\n')}\n\n`;
            }
            
            if (detailedAnalysis.growth_areas.development_path) {
              formattedContent += `Development Path:\n${detailedAnalysis.growth_areas.development_path}\n\n`;
            }
          }
        });
        
        // Add group dynamics at the end
        if (personalityInsights.groupDynamics) {
          formattedContent += `${'─'.repeat(65)}\n`;
          formattedContent += `🤝 Group Dynamics (${peopleCount}-Person Analysis)\n`;
          formattedContent += `${'─'.repeat(65)}\n\n`;
          formattedContent += `${personalityInsights.groupDynamics}\n`;
        }
        
      } else if (personalityInsights.individualProfiles?.length === 1) {
        // Single person format with 20 core psychological questions
        const profile = personalityInsights.individualProfiles[0];
        const detailedAnalysis = profile.detailed_analysis || {};
        const coreAssessment = detailedAnalysis.core_psychological_assessment || {};
        
        const gender = profile.personLabel?.includes('Male') ? 'Male' : 
                       profile.personLabel?.includes('Female') ? 'Female' : '';
        const ageMatch = profile.personLabel?.match(/~(\d+)-(\d+)/);
        const ageRange = ageMatch ? `~${ageMatch[1]}–${ageMatch[2]} years` : '';
        const genderAge = [gender, ageRange].filter(Boolean).join(', ');
        
        formattedContent = `AI-Powered Psychological Profile Report\n`;
        formattedContent += `Subject Detected: 1 Individual\n`;
        formattedContent += `Mode: Individual Analysis\n\n`;
        
        formattedContent += `Subject 1${genderAge ? ` (${genderAge})` : ''}\n`;
        formattedContent += `${'─'.repeat(40)}\n\n`;
        
        formattedContent += `Summary:\n${profile.summary || 'No summary available'}\n\n`;
        
        // Display the 20 Core Psychological Questions
        formattedContent += `Core Psychological Assessment\n\n`;
        
        formattedContent += `What drives this person: ${coreAssessment.core_motivation || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Confidence level: ${coreAssessment.confidence_level || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Self-acceptance: ${coreAssessment.self_acceptance || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Intelligence level: ${coreAssessment.intelligence_level || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Creativity: ${coreAssessment.creativity_assessment || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Stress handling: ${coreAssessment.stress_handling || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Trustworthiness: ${coreAssessment.trustworthiness || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Authenticity: ${coreAssessment.authenticity || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Ambition level: ${coreAssessment.ambition_level || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Insecurities: ${coreAssessment.insecurities || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Social validation needs: ${coreAssessment.social_validation || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Independence: ${coreAssessment.independence || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Communication style: ${coreAssessment.communication_style || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Response to criticism: ${coreAssessment.criticism_response || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Outlook: ${coreAssessment.outlook || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Sense of humor: ${coreAssessment.humor_sense || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Treatment of others: ${coreAssessment.treatment_of_others || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Consistency: ${coreAssessment.consistency || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Hidden strengths: ${coreAssessment.hidden_strengths || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        formattedContent += `Hidden weaknesses: ${coreAssessment.hidden_weaknesses || '[ERROR: AI model failed to provide answer - please regenerate analysis]'}\n\n`;
        
        // Speech Analysis section if available
        if (detailedAnalysis.speech_analysis) {
          formattedContent += `Speech Analysis & Quotes\n`;
          const speechAnalysis = detailedAnalysis.speech_analysis;
          
          if (speechAnalysis.key_quotes && Array.isArray(speechAnalysis.key_quotes) && speechAnalysis.key_quotes.length > 0) {
            formattedContent += `Key Quotes: ${speechAnalysis.key_quotes.join(' | ')}\n\n`;
          }
          
          if (speechAnalysis.vocabulary_analysis) {
            formattedContent += `Vocabulary Analysis: ${speechAnalysis.vocabulary_analysis}\n\n`;
          }
          
          if (speechAnalysis.personality_revealed) {
            formattedContent += `Personality Revealed: ${speechAnalysis.personality_revealed}\n\n`;
          }
        }
        
        // Visual Evidence section if available  
        if (detailedAnalysis.visual_evidence) {
          formattedContent += `Visual Evidence\n`;
          const visualEvidence = detailedAnalysis.visual_evidence;
          
          if (visualEvidence.facial_analysis) {
            formattedContent += `Facial Analysis: ${visualEvidence.facial_analysis}\n\n`;
          }
          
          if (visualEvidence.body_language) {
            formattedContent += `Body Language: ${visualEvidence.body_language}\n\n`;
          }
          
          if (visualEvidence.appearance_details) {
            formattedContent += `Appearance Details: ${visualEvidence.appearance_details}\n\n`;
          }
          
          if (visualEvidence.microexpressions) {
            formattedContent += `Microexpressions: ${visualEvidence.microexpressions}\n\n`;
          }
        }
        
        // Professional Insights
        if (detailedAnalysis.professional_insights) {
          formattedContent += `Professional Insights\n${detailedAnalysis.professional_insights}\n\n`;
        }
        
        // Growth Areas
        if (detailedAnalysis.growth_areas) {
          formattedContent += `Growth Areas\n`;
          
          if (Array.isArray(detailedAnalysis.growth_areas.strengths) && 
              detailedAnalysis.growth_areas.strengths.length > 0) {
            formattedContent += `Strengths: ${detailedAnalysis.growth_areas.strengths.join(', ')}\n\n`;
          }
          
          if (detailedAnalysis.growth_areas.development_path) {
            formattedContent += `Development Path: ${detailedAnalysis.growth_areas.development_path}\n\n`;
          }
        }
      } else {
        // Fallback if no profiles
        formattedContent = "No personality profiles could be generated. Please try again with a different image or video.";
      }

      // Send initial message with comprehensive analysis
      const message = await storage.createMessage({
        sessionId,
        analysisId: analysis.id,
        content: formattedContent,
        role: "assistant",
      });

      // Get all messages to return to client
      const messages = await storage.getMessagesBySessionId(sessionId);

      res.json({ 
        ...analysis, 
        messages,
        emailServiceAvailable: isEmailServiceConfigured 
      });
      
      console.log(`Analysis complete. Created message with ID ${message.id} and returning ${messages.length} messages`);
    } catch (error) {
      console.error("Analyze error:", error);
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(400).json({ error: "An unknown error occurred" });
      }
    }
  });

  app.post("/api/chat", async (req, res) => {
    try {
      const { content, sessionId } = sendMessageSchema.parse(req.body);

      const userMessage = await storage.createMessage({
        sessionId,
        content,
        role: "user",
      });

      // Check if OpenAI client is available
      if (!openai) {
        return res.status(400).json({ 
          error: "OpenAI API key is not configured. Please provide an OpenAI API key to use the chat functionality.",
          configError: "OPENAI_API_KEY_MISSING",
          messages: [userMessage]
        });
      }

      const analysis = await storage.getAnalysisBySessionId(sessionId);
      const messages = await storage.getMessagesBySessionId(sessionId);

      try {
        // Set up the messages for the API call
        const apiMessages = [
          {
            role: "system",
            content: `You are an AI assistant capable of general conversation as well as providing specialized analysis about the personality insights previously generated. 
            
If the user asks about the analysis, provide detailed information based on the personality insights.
If the user asks general questions unrelated to the analysis, respond naturally and helpfully as you would to any question.

IMPORTANT: Do not use markdown formatting in your responses. Do not use ** for bold text, do not use ### for headers, and do not use markdown formatting for bullet points or numbered lists. Use plain text formatting only.

Be engaging, professional, and conversational in all responses. Feel free to have opinions, share information, and engage in dialogue on any topic.`,
          },
          {
            role: "assistant",
            content: typeof analysis?.personalityInsights === 'object' 
              ? JSON.stringify(analysis?.personalityInsights) 
              : String(analysis?.personalityInsights || ''),
          },
          ...messages.map(m => ({ role: m.role, content: m.content })),
        ];
        
        // Convert message format to match OpenAI's expected types
        const typedMessages = apiMessages.map(msg => {
          // Convert role to proper type
          const role = msg.role === 'user' ? 'user' : 
                      msg.role === 'assistant' ? 'assistant' : 'system';
          
          // Return properly typed message
          return {
            role,
            content: msg.content || ''
          };
        });
        
        // Use the properly typed messages for the API call
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: typedMessages,
          // Don't use JSON format as it requires specific message formats
          // response_format: { type: "json_object" },
        });

        // Get the raw text response
        const responseContent = response.choices[0]?.message.content || "";
        let aiResponse = responseContent;
        
        // Try to parse as JSON if it appears to be JSON, otherwise use as plain text
        try {
          if (responseContent.trim().startsWith('{') && responseContent.trim().endsWith('}')) {
            aiResponse = JSON.parse(responseContent);
          }
        } catch (e) {
          // If parsing fails, use the raw text
          console.log("Failed to parse response as JSON, using raw text");
          aiResponse = responseContent;
        }

        // Create the assistant message using the response content
        // If aiResponse is an object with a response property, use that
        // Otherwise, use the raw text response
        const messageContent = typeof aiResponse === 'object' && aiResponse.response 
          ? aiResponse.response 
          : typeof aiResponse === 'string' 
            ? aiResponse 
            : "I'm sorry, I couldn't generate a proper response.";
            
        const assistantMessage = await storage.createMessage({
          sessionId,
          analysisId: analysis?.id,
          content: messageContent,
          role: "assistant",
        });

        res.json({ messages: [userMessage, assistantMessage] });
      } catch (apiError) {
        console.error("OpenAI API error:", apiError);
        res.status(500).json({ 
          error: "Error communicating with OpenAI API. Please check your API key configuration.",
          messages: [userMessage]
        });
      }
    } catch (error) {
      console.error("Chat processing error:", error);
      res.status(400).json({ error: "Failed to process chat message" });
    }
  });

  app.get("/api/messages", async (req, res) => {
    try {
      const { sessionId } = req.query;
      
      if (!sessionId || typeof sessionId !== 'string') {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      const messages = await storage.getMessagesBySessionId(sessionId);
      res.json(messages);
    } catch (error) {
      console.error("Get messages error:", error);
      res.status(400).json({ error: "Failed to get messages" });
    }
  });

  app.get("/api/shared-analysis/:shareId", async (req, res) => {
    try {
      const { shareId } = getSharedAnalysisSchema.parse({ shareId: req.params.shareId });
      
      // Get the share record
      const share = await storage.getShareById(shareId);
      if (!share) {
        return res.status(404).json({ error: "Shared analysis not found" });
      }
      
      // Get the analysis
      const analysis = await storage.getAnalysisById(share.analysisId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      
      // Get all messages for this analysis
      const messages = await storage.getMessagesBySessionId(analysis.sessionId);
      
      // Return the complete data
      res.json({
        analysis,
        messages,
        share,
        emailServiceAvailable: isEmailServiceConfigured
      });
    } catch (error) {
      console.error("Get shared analysis error:", error);
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(400).json({ error: "Failed to get shared analysis" });
      }
    }
  });

  // API status endpoint - returns the availability of various services
  app.get("/api/status", async (req, res) => {
    try {
      const statusData = {
        // LLM Services
        openai: !!openai,
        anthropic: !!anthropic,
        perplexity: !!process.env.PERPLEXITY_API_KEY,
        azureOpenai: !!process.env.AZURE_OPENAI_ENDPOINT && !!process.env.AZURE_OPENAI_KEY,
        
        // Facial Analysis Services
        aws_rekognition: !!process.env.AWS_ACCESS_KEY_ID && !!process.env.AWS_SECRET_ACCESS_KEY,
        facepp: !!process.env.FACEPP_API_KEY && !!process.env.FACEPP_API_SECRET,
        azure_face: !!process.env.AZURE_FACE_ENDPOINT && !!process.env.AZURE_FACE_API_KEY,
        google_vision: !!process.env.GOOGLE_CLOUD_VISION_API_KEY,
        
        // Transcription Services
        gladia: !!process.env.GLADIA_API_KEY,
        assemblyai: !!process.env.ASSEMBLYAI_API_KEY,
        deepgram: !!process.env.DEEPGRAM_API_KEY,
        
        // Video Analysis Services
        azure_video_indexer: !!process.env.AZURE_VIDEO_INDEXER_KEY && 
                            !!process.env.AZURE_VIDEO_INDEXER_LOCATION && 
                            !!process.env.AZURE_VIDEO_INDEXER_ACCOUNT_ID,
        
        // Email Service
        sendgrid: !!process.env.SENDGRID_API_KEY && !!process.env.SENDGRID_VERIFIED_SENDER,
        
        // Service status timestamp
        timestamp: new Date().toISOString()
      };
      
      res.json(statusData);
    } catch (error) {
      console.error("Error checking API status:", error);
      res.status(500).json({ error: "Failed to check API status" });
    }
  });
  
  // Session management endpoints
  app.get("/api/sessions", async (req, res) => {
    try {
      const sessions = await storage.getAllSessions();
      res.json(sessions);
    } catch (error) {
      console.error("Error getting sessions:", error);
      res.status(500).json({ error: "Failed to get sessions" });
    }
  });
  
  app.post("/api/session/clear", async (req, res) => {
    try {
      const { sessionId } = req.body;
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      await storage.clearSession(sessionId);
      res.json({ success: true });
    } catch (error) {
      console.error("Error clearing session:", error);
      res.status(500).json({ error: "Failed to clear session" });
    }
  });
  
  app.patch("/api/session/name", async (req, res) => {
    try {
      const { sessionId, name } = req.body;
      
      if (!sessionId) {
        return res.status(400).json({ error: "Session ID is required" });
      }
      
      if (!name) {
        return res.status(400).json({ error: "Name is required" });
      }
      
      await storage.updateSessionName(sessionId, name);
      res.json({ success: true });
    } catch (error) {
      console.error("Error updating session name:", error);
      res.status(500).json({ error: "Failed to update session name" });
    }
  });
  
  // Test email endpoint (for troubleshooting only, disable in production)
  app.get("/api/test-email", async (req, res) => {
    try {
      if (!process.env.SENDGRID_API_KEY || !process.env.SENDGRID_VERIFIED_SENDER) {
        return res.status(503).json({ 
          error: "Email service is not available. Please check environment variables." 
        });
      }
      
      // Create a test share
      const testShare = {
        id: 9999,
        analysisId: 9999,
        senderEmail: "test@example.com",
        recipientEmail: process.env.SENDGRID_VERIFIED_SENDER, // Use the verified sender as recipient for testing
        status: "pending",
        createdAt: new Date().toISOString()
      };
      
      // Create a test analysis
      const testAnalysis = {
        id: 9999,
        sessionId: "test-session",
        title: "Test Analysis",
        mediaType: "text",
        mediaUrl: null,
        peopleCount: 1,
        personalityInsights: {
          summary: "This is a test analysis summary for email testing purposes.",
          personality_core: {
            summary: "Test personality core summary."
          },
          thought_patterns: {
            summary: "Test thought patterns summary."
          },
          professional_insights: {
            summary: "Test professional insights summary."
          },
          growth_areas: {
            strengths: ["Test strength 1", "Test strength 2"],
            challenges: ["Test challenge 1", "Test challenge 2"],
            development_path: "Test development path."
          }
        },
        downloaded: false,
        createdAt: new Date().toISOString()
      };
      
      // Send test email
      console.log("Sending test email...");
      const emailSent = await sendAnalysisEmail({
        share: testShare,
        analysis: testAnalysis,
        shareUrl: "https://example.com/test-share"
      });
      
      if (emailSent) {
        res.json({ success: true, message: "Test email sent successfully" });
      } else {
        res.status(500).json({ success: false, error: "Failed to send test email" });
      }
    } catch (error) {
      console.error("Test email error:", error);
      res.status(500).json({ success: false, error: String(error) });
    }
  });
  
  // Get a specific analysis by ID
  app.get("/api/analysis/:id", async (req, res) => {
    try {
      const analysisId = parseInt(req.params.id);
      if (isNaN(analysisId)) {
        return res.status(400).json({ error: 'Invalid analysis ID' });
      }
      
      const analysis = await storage.getAnalysisById(analysisId);
      if (!analysis) {
        return res.status(404).json({ error: 'Analysis not found' });
      }
      
      res.json(analysis);
    } catch (error) {
      console.error('Error fetching analysis:', error);
      res.status(500).json({ error: 'Failed to fetch analysis' });
    }
  });
  
  // Download analysis as PDF or DOCX
  app.get("/api/download/:analysisId", async (req, res) => {
    try {
      const { analysisId } = req.params;
      const format = req.query.format as string || 'pdf';
      
      // Get the analysis from storage
      const analysis = await storage.getAnalysisById(parseInt(analysisId));
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }
      
      let buffer: Buffer;
      let contentType: string;
      let filename: string;
      
      if (format === 'docx') {
        // Generate DOCX
        buffer = await generateDocx(analysis);
        contentType = 'application/vnd.openxmlformats-officedocument.wordprocessingml.document';
        filename = `personality-analysis-${analysisId}.docx`;
      } else if (format === 'txt') {
        // Generate TXT
        const txtContent = generateAnalysisTxt(analysis);
        buffer = Buffer.from(txtContent, 'utf-8');
        contentType = 'text/plain';
        filename = `personality-analysis-${analysisId}.txt`;
      } else {
        // Default to PDF
        const htmlContent = generateAnalysisHtml(analysis);
        buffer = await generatePdf(htmlContent);
        contentType = 'application/pdf';
        filename = `personality-analysis-${analysisId}.pdf`;
      }
      
      // Mark as downloaded in the database
      await storage.updateAnalysisDownloadStatus(analysis.id, true);
      
      // Send the file
      res.setHeader('Content-Type', contentType);
      res.setHeader('Content-Disposition', `attachment; filename="${filename}"`);
      res.setHeader('Content-Length', buffer.length);
      res.send(buffer);
      
    } catch (error) {
      console.error("Download error:", error);
      res.status(500).json({ error: "Failed to generate document" });
    }
  });

  app.post("/api/share", async (req, res) => {
    try {
      // Check if email service is configured
      if (!isEmailServiceConfigured) {
        return res.status(503).json({ 
          error: "Email sharing is not available. Please try again later or contact support." 
        });
      }

      const shareData = insertShareSchema.parse(req.body);

      // Create share record
      const share = await storage.createShare(shareData);

      // Get the analysis
      const analysis = await storage.getAnalysisById(shareData.analysisId);
      if (!analysis) {
        return res.status(404).json({ error: "Analysis not found" });
      }

      // Generate the share URL with the current hostname and /share path with analysis ID
      const hostname = req.get('host');
      const protocol = req.headers['x-forwarded-proto'] || req.protocol;
      const shareUrl = `${protocol}://${hostname}/share/${share.id}`;
      
      // Send email with share URL
      const emailSent = await sendAnalysisEmail({
        share,
        analysis,
        shareUrl
      });

      // Update share status based on email sending result
      await storage.updateShareStatus(share.id, emailSent ? "sent" : "error");

      if (!emailSent) {
        return res.status(500).json({ 
          error: "Failed to send email. Please try again later." 
        });
      }

      res.json({ success: emailSent, shareUrl });
    } catch (error) {
      console.error('Share endpoint error:', error);
      if (error instanceof Error) {
        res.status(400).json({ error: error.message });
      } else {
        res.status(400).json({ error: "Failed to share analysis" });
      }
    }
  });

  const httpServer = createServer(app);
  return httpServer;
}

/**
 * Enhanced face analysis function that uses multiple services with fallback
 * Tries Azure Face API first, then Face++, and finally AWS Rekognition
 */
async function analyzeFaces(imageBuffer: Buffer, maxPeople: number = 5) {
  let analysisResult = {
    provider: "none",
    faces: [] as any[],
    success: false
  };
  
  // First try Azure Face API if available
  if (AZURE_FACE_API_KEY && AZURE_FACE_ENDPOINT) {
    try {
      console.log('Attempting face analysis with Azure Face API...');
      
      // Update: Use the modern Azure Face API without deprecated attributes
      // The newer version doesn't support emotion detection and some other attributes that were deprecated
      const azureResponse = await fetch(`${AZURE_FACE_ENDPOINT}/face/v1.0/detect?returnFaceId=true&returnFaceLandmarks=true&recognitionModel=recognition_04`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/octet-stream',
          'Ocp-Apim-Subscription-Key': AZURE_FACE_API_KEY
        },
        body: imageBuffer
      });
      
      if (azureResponse.ok) {
        const facesData = await azureResponse.json() as any;
        
        if (facesData && Array.isArray(facesData) && facesData.length > 0) {
          // Process and format the Azure response
          const processedFaces = facesData.slice(0, maxPeople).map((face: any, index: number) => {
            // Create descriptive label for this person (without gender/age which are no longer available)
            const personLabel = `Person ${index + 1}`;
            
            // Get face size and position
            const faceWidth = face.faceRectangle.width;
            const faceHeight = face.faceRectangle.height;
            const faceArea = faceWidth * faceHeight;
            
            // Create a normalized bounding box (0-1 range)
            // Assuming the image dimensions based on the face position
            const imageWidth = Math.max(1000, face.faceRectangle.left + face.faceRectangle.width * 2);
            const imageHeight = Math.max(1000, face.faceRectangle.top + face.faceRectangle.height * 2);
            
            const boundingBox = {
              Width: face.faceRectangle.width / imageWidth,
              Height: face.faceRectangle.height / imageHeight,
              Left: face.faceRectangle.left / imageWidth,
              Top: face.faceRectangle.top / imageHeight
            };
            
            // Estimate age range (since no longer provided by the API)
            const estimatedAge = {
              low: 20,
              high: 40
            };
            
            // Use face landmarks to estimate expressions and attributes
            const landmarks = face.faceLandmarks;
            
            // Calculate approximate smile score based on landmarks
            let smileScore = 0;
            
            if (landmarks) {
              // Estimate smile by looking at mouth corners relative to mouth center
              const mouthLeft = landmarks.mouthLeft;
              const mouthRight = landmarks.mouthRight; 
              const upperLipTop = landmarks.upperLipTop;
              
              if (mouthLeft && mouthRight && upperLipTop) {
                // Simple smile detection based on mouth curve
                // If mouth corners are higher than the center, it might indicate a smile
                const mouthCurve = ((mouthLeft.y + mouthRight.y) / 2) - upperLipTop.y;
                smileScore = Math.max(0, Math.min(1, mouthCurve / 10));
              }
            }
            
            // Estimate basic emotions (simplified since these are no longer provided by the API)
            const estimatedEmotions = {
              neutral: 0.7,
              happiness: smileScore
            };
            
            return {
              personLabel,
              positionInImage: index + 1,
              boundingBox,
              age: estimatedAge,
              gender: "unknown", // No longer provided by Azure Face API
              emotion: estimatedEmotions,
              faceAttributes: {
                smile: smileScore,
                eyeglasses: "Unknown", // No longer provided by Azure Face API
                sunglasses: "Unknown", // No longer provided by Azure Face API
                beard: "Unknown", // No longer provided by Azure Face API
                mustache: "Unknown", // No longer provided by Azure Face API
                eyesOpen: "Unknown", // Not directly provided by Azure
                mouthOpen: "Unknown", // Not directly provided by Azure
                quality: {
                  brightness: 0, // Not directly provided by Azure in new API
                  sharpness: 0 // Not directly provided by Azure in new API
                },
                pose: {
                  pitch: 0, // Not directly provided by Azure
                  roll: 0,
                  yaw: 0
                }
              },
              dominant: index === 0
            };
          });
          
          analysisResult = {
            provider: "azure",
            faces: processedFaces,
            success: true
          };
          
          console.log('Azure Face API analysis successful!');
          return processedFaces;
        }
      } else {
        console.warn('Azure Face API returned an error:', await azureResponse.text());
      }
    } catch (error) {
      console.error('Azure Face API analysis error:', error);
    }
  }
  
  // If Azure failed, try Face++ if available
  if (!analysisResult.success && FACEPP_API_KEY && FACEPP_API_SECRET) {
    try {
      console.log('Attempting face analysis with Face++ API...');
      
      // Format the image data for Face++ API
      const formData = new FormData();
      formData.append('api_key', FACEPP_API_KEY);
      formData.append('api_secret', FACEPP_API_SECRET);
      formData.append('image_file', imageBuffer, 'image.jpg');
      formData.append('return_landmark', '0');
      formData.append('return_attributes', 'gender,age,smiling,emotion,eyestatus,mouthstatus,eyegaze,beauty,skinstatus');
      
      // @ts-ignore: FormData is compatible with fetch API's Body type
      const faceppResponse = await fetch('https://api-us.faceplusplus.com/facepp/v3/detect', {
        method: 'POST',
        body: formData
      });
      
      if (faceppResponse.ok) {
        const facesData = await faceppResponse.json() as any;
        
        if (facesData && facesData.faces && facesData.faces.length > 0) {
          // Process and format the Face++ response
          const processedFaces = facesData.faces.slice(0, maxPeople).map((face: any, index: number) => {
            // Create descriptive label
            const genderValue = face.attributes?.gender?.value || 'unknown';
            const genderLabel = genderValue.toLowerCase() === 'male' ? 'Male' : 'Female';
            const ageValue = face.attributes?.age?.value || 0;
            const personLabel = `Person ${index + 1} (${genderLabel}, ~${ageValue} years)`;
            
            // Map emotions to standardized format
            const emotions = face.attributes?.emotion || {};
            const emotionMap: Record<string, number> = {};
            
            Object.keys(emotions).forEach(emotion => {
              emotionMap[emotion.toLowerCase()] = emotions[emotion] / 100;
            });
            
            const faceRect = face.face_rectangle || {};
            
            return {
              personLabel,
              positionInImage: index + 1,
              boundingBox: {
                Width: faceRect.width / 100,
                Height: faceRect.height / 100,
                Left: faceRect.left / 100,
                Top: faceRect.top / 100
              },
              age: {
                low: Math.max(0, ageValue - 5),
                high: ageValue + 5
              },
              gender: genderValue.toLowerCase(),
              emotion: emotionMap,
              faceAttributes: {
                smile: face.attributes?.smile?.value / 100 || 0,
                eyeglasses: face.attributes?.eyeglass?.value > 50 ? "Glasses" : "NoGlasses",
                sunglasses: face.attributes?.sunglass?.value > 50 ? "Sunglasses" : "NoSunglasses",
                beard: face.attributes?.beard?.value > 50 ? "Yes" : "No",
                mustache: face.attributes?.moustache?.value > 50 ? "Yes" : "No",
                eyesOpen: face.attributes?.eyestatus?.left_eye_status?.eye_open > 50 ? "Yes" : "No",
                mouthOpen: face.attributes?.mouthstatus?.open > 50 ? "Yes" : "No",
                quality: {
                  brightness: 0, // Not directly provided
                  sharpness: 0, // Not directly provided
                },
                pose: {
                  pitch: 0, // Not directly provided in basic mode
                  roll: 0,  // Not directly provided in basic mode
                  yaw: 0    // Not directly provided in basic mode
                }
              },
              dominant: index === 0
            };
          });
          
          analysisResult = {
            provider: "facepp",
            faces: processedFaces,
            success: true
          };
          
          console.log('Face++ API analysis successful!');
          return processedFaces;
        }
      } else {
        console.warn('Face++ API returned an error:', await faceppResponse.text());
      }
    } catch (error) {
      console.error('Face++ API analysis error:', error);
    }
  }
  
  // Try Google Cloud Vision if previous methods failed and API key is available
  if (!analysisResult.success && GOOGLE_CLOUD_VISION_API_KEY) {
    try {
      console.log('Attempting face analysis with Google Cloud Vision API...');
      
      // Prepare the request to Google Cloud Vision API
      const requestBody = {
        requests: [
          {
            image: {
              content: imageBuffer.toString('base64')
            },
            features: [
              {
                type: "FACE_DETECTION",
                maxResults: maxPeople
              }
            ]
          }
        ]
      };
      
      const gcvResponse = await fetch(`https://vision.googleapis.com/v1/images:annotate?key=${GOOGLE_CLOUD_VISION_API_KEY}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      });
      
      if (gcvResponse.ok) {
        const result = await gcvResponse.json();
        const faceAnnotations = result.responses?.[0]?.faceAnnotations || [];
        
        if (faceAnnotations.length > 0) {
          // Process and format the Google Cloud Vision response
          const processedFaces = faceAnnotations.slice(0, maxPeople).map((face: any, index: number) => {
            // Create a descriptive label for each person
            const personLabel = `Person ${index + 1}`;
            
            // Get vertices of the face bounding polygon
            const vertices = face.boundingPoly?.vertices || [];
            let left = 0, top = 0, right = 0, bottom = 0;
            
            if (vertices.length >= 4) {
              left = Math.min(...vertices.map((v: any) => v.x || 0));
              top = Math.min(...vertices.map((v: any) => v.y || 0));
              right = Math.max(...vertices.map((v: any) => v.x || 0));
              bottom = Math.max(...vertices.map((v: any) => v.y || 0));
            }
            
            // Create normalized bounding box (0-1 range)
            // Note: We're estimating the image size from the face bounds
            const imageWidth = 1000; // Approximate width for normalization
            const imageHeight = 1000; // Approximate height for normalization
            
            const boundingBox = {
              Width: (right - left) / imageWidth,
              Height: (bottom - top) / imageHeight,
              Left: left / imageWidth,
              Top: top / imageHeight
            };
            
            // Map Google Cloud Vision emotions to our standard format
            const emotionMap: Record<string, number> = {
              joy: face.joyLikelihood === "VERY_LIKELY" ? 0.9 : 
                   face.joyLikelihood === "LIKELY" ? 0.7 :
                   face.joyLikelihood === "POSSIBLE" ? 0.5 :
                   face.joyLikelihood === "UNLIKELY" ? 0.3 : 0.1,
              sorrow: face.sorrowLikelihood === "VERY_LIKELY" ? 0.9 : 
                      face.sorrowLikelihood === "LIKELY" ? 0.7 :
                      face.sorrowLikelihood === "POSSIBLE" ? 0.5 :
                      face.sorrowLikelihood === "UNLIKELY" ? 0.3 : 0.1,
              anger: face.angerLikelihood === "VERY_LIKELY" ? 0.9 : 
                     face.angerLikelihood === "LIKELY" ? 0.7 :
                     face.angerLikelihood === "POSSIBLE" ? 0.5 :
                     face.angerLikelihood === "UNLIKELY" ? 0.3 : 0.1,
              surprise: face.surpriseLikelihood === "VERY_LIKELY" ? 0.9 : 
                        face.surpriseLikelihood === "LIKELY" ? 0.7 :
                        face.surpriseLikelihood === "POSSIBLE" ? 0.5 :
                        face.surpriseLikelihood === "UNLIKELY" ? 0.3 : 0.1
            };
            
            return {
              personLabel,
              positionInImage: index + 1,
              boundingBox,
              age: {
                low: 18, // GCV doesn't provide exact age estimates
                high: 50
              },
              gender: "unknown", // GCV doesn't provide gender
              emotion: emotionMap,
              faceAttributes: {
                smile: emotionMap.joy,
                eyeglasses: "Unknown", // Not provided
                sunglasses: "Unknown", // Not provided
                beard: "Unknown", // Not provided
                mustache: "Unknown", // Not provided
                eyesOpen: "Unknown", // Not provided
                mouthOpen: "Unknown", // Not provided
                quality: {
                  brightness: 0, // Not directly provided
                  sharpness: 0 // Not directly provided
                },
                pose: {
                  pitch: face.tiltAngle || 0,
                  roll: face.rollAngle || 0,
                  yaw: face.panAngle || 0
                }
              },
              dominant: index === 0
            };
          });
          
          analysisResult = {
            provider: "google_cloud_vision",
            faces: processedFaces,
            success: true
          };
          
          console.log('Google Cloud Vision face analysis successful!');
          return processedFaces;
        }
      } else {
        console.warn('Google Cloud Vision API returned an error:', await gcvResponse.text());
      }
    } catch (error) {
      console.error('Google Cloud Vision analysis error:', error);
    }
  }
  
  // If all previous methods failed, fall back to AWS Rekognition
  try {
    console.log('Falling back to AWS Rekognition for face analysis...');
    
    const command = new DetectFacesCommand({
      Image: {
        Bytes: imageBuffer
      },
      Attributes: ['ALL']
    });

    const response = await rekognition.send(command);
    const faces = response.FaceDetails || [];

    if (faces.length === 0) {
      throw new Error("No faces detected in the image");
    }

    // Limit the number of faces to analyze
    const facesToProcess = faces.slice(0, maxPeople);
    
    // Process each face and add descriptive labels
    const processedFaces = facesToProcess.map((face, index) => {
      // Create a descriptive label for each person
      let personLabel = `Person ${index + 1}`;
      
      // Add gender and approximate age to label if available
      if (face.Gender?.Value) {
        const genderLabel = face.Gender.Value.toLowerCase() === 'male' ? 'Male' : 'Female';
        const ageRange = face.AgeRange ? `${face.AgeRange.Low}-${face.AgeRange.High}` : '';
        personLabel = `${personLabel} (${genderLabel}${ageRange ? ', ~' + ageRange + ' years' : ''})`;
      }
    
      return {
        personLabel,
        positionInImage: index + 1,
        boundingBox: face.BoundingBox || {
          Width: 0,
          Height: 0,
          Left: 0,
          Top: 0
        },
        age: {
          low: face.AgeRange?.Low || 0,
          high: face.AgeRange?.High || 0
        },
        gender: face.Gender?.Value?.toLowerCase() || "unknown",
        emotion: face.Emotions?.reduce((acc, emotion) => {
          if (emotion.Type && emotion.Confidence) {
            acc[emotion.Type.toLowerCase()] = emotion.Confidence / 100;
          }
          return acc;
        }, {} as Record<string, number>),
        faceAttributes: {
          smile: face.Smile?.Value ? (face.Smile.Confidence || 0) / 100 : 0,
          eyeglasses: face.Eyeglasses?.Value ? "Glasses" : "NoGlasses",
          sunglasses: face.Sunglasses?.Value ? "Sunglasses" : "NoSunglasses",
          beard: face.Beard?.Value ? "Yes" : "No",
          mustache: face.Mustache?.Value ? "Yes" : "No",
          eyesOpen: face.EyesOpen?.Value ? "Yes" : "No",
          mouthOpen: face.MouthOpen?.Value ? "Yes" : "No",
          quality: {
            brightness: face.Quality?.Brightness || 0,
            sharpness: face.Quality?.Sharpness || 0,
          },
          pose: {
            pitch: face.Pose?.Pitch || 0,
            roll: face.Pose?.Roll || 0,
            yaw: face.Pose?.Yaw || 0,
          }
        },
        dominant: index === 0 // Flag the first/largest face as dominant
      };
    });
    
    analysisResult = {
      provider: "aws_rekognition",
      faces: processedFaces,
      success: true
    };
    
    console.log('AWS Rekognition face analysis successful!');
    return processedFaces;
  } catch (error) {
    console.error('AWS Rekognition analysis failed:', error);
    
    // If all face detection methods fail, throw an error
    if (!analysisResult.success) {
      throw new Error("No faces detected in the image by any provider");
    }
    
    // Return any successful results from previous providers
    return analysisResult.faces;
  }
}

// For backward compatibility
async function analyzeFaceWithRekognition(imageBuffer: Buffer, maxPeople: number = 5) {
  return analyzeFaces(imageBuffer, maxPeople);
}



/**
 * Validates that all 20 core psychological questions are answered with substantive content
 * Throws error if validation fails to force regeneration
 */
function validateCoreAssessment(analysisResult: any, personLabel: string = "Subject") {
  const coreQuestions = [
    'core_motivation', 'confidence_level', 'self_acceptance', 'intelligence_level',
    'creativity_assessment', 'stress_handling', 'trustworthiness', 'authenticity',
    'ambition_level', 'insecurities', 'social_validation', 'independence',
    'communication_style', 'criticism_response', 'outlook', 'humor_sense',
    'treatment_of_others', 'consistency', 'hidden_strengths', 'hidden_weaknesses'
  ];
  
  const coreAssessment = analysisResult.detailed_analysis?.core_psychological_assessment;
  
  if (!coreAssessment) {
    throw new Error(`AI model failed to provide core psychological assessment for ${personLabel}. Please regenerate the analysis.`);
  }
  
  const missingFields = [];
  for (const question of coreQuestions) {
    if (!coreAssessment[question] || coreAssessment[question].trim().length < 10) {
      missingFields.push(question);
    }
  }
  
  if (missingFields.length > 0) {
    console.error(`Validation failed for ${personLabel}! Missing or incomplete answers for: ${missingFields.join(', ')}`);
    throw new Error(`AI model provided incomplete analysis for ${personLabel}. Missing substantive answers for: ${missingFields.join(', ')}. Please regenerate the analysis with a different model or try again.`);
  }
  
  console.log(`✅ Validation passed for ${personLabel} - all 20 core questions answered`);
  return true;
}

async function getEnhancedPersonalityInsights(faceAnalysis: any, videoAnalysis: any = null, audioTranscription: any = null, selectedModel: string = "deepseek") {
  // Check if any API clients are available, display warning if not
  if (!deepseek && !openai && !anthropic && !process.env.PERPLEXITY_API_KEY) {
    console.warn("No AI model API clients are available. Using fallback analysis.");
    return {
      peopleCount: Array.isArray(faceAnalysis) ? faceAnalysis.length : 1,
      individualProfiles: [{
        summary: "API keys are required for detailed analysis. Please configure OpenAI, Anthropic, or Perplexity API keys.",
        detailed_analysis: {
          personality_core: "API keys required for detailed analysis",
          thought_patterns: "API keys required for detailed analysis",
          cognitive_style: "API keys required for detailed analysis",
          professional_insights: "API keys required for detailed analysis",
          relationships: {
            current_status: "Not available",
            parental_status: "Not available",
            ideal_partner: "Not available"
          },
          growth_areas: {
            strengths: ["Not available"],
            challenges: ["Not available"],
            development_path: "Not available"
          }
        }
      }]
    };
  }
  
  // Check if faceAnalysis is an array (multiple people) or single object
  const isMultiplePeople = Array.isArray(faceAnalysis);
  
  // If we have multiple people, analyze each one separately
  if (isMultiplePeople) {
    console.log(`Analyzing ${faceAnalysis.length} people...`);
    
    // Create a combined analysis with an overview and individual profiles
    let multiPersonAnalysis = {
      peopleCount: faceAnalysis.length,
      overviewSummary: `Analysis of ${faceAnalysis.length} people detected in the media.`,
      individualProfiles: [] as any[],
      groupDynamics: undefined as string | undefined, // Will be populated later for multi-person analyses
      detailed_analysis: {} // For backward compatibility with message format
    };
    
    // Analyze each person with the existing logic (concurrently for efficiency)
    const analysisPromises = faceAnalysis.map(async (personFaceData) => {
      try {
        // Create input for this specific person
        const personInput = {
          faceAnalysis: personFaceData,
          ...(videoAnalysis && { videoAnalysis }),
          ...(audioTranscription && { 
            audioTranscription: {
              ...audioTranscription,
              // Ensure we're passing the structured transcription data for quote extraction
              transcriptionData: audioTranscription.transcriptionData || {
                full_text: audioTranscription.transcription || "",
                utterances: [],
                words: []
              }
            } 
          })
        };
        
        // Use the standard analysis prompt but customized for the person
        const personLabel = personFaceData.personLabel || "Person";
        const analysisPrompt = `
You are an expert psychologist, cognitive scientist, and personality analyst with deep expertise in psychological assessment and cognitive profiling. 
Conduct an ENHANCED, COMPREHENSIVE, and DEEP psychological analysis for ${personLabel}. This should be a thorough, professional-grade assessment with extensive detail and insights.

CRITICAL REQUIREMENT: YOU MUST START WITH DETAILED VISUAL DESCRIPTION. Begin your analysis by describing what you actually see in the image/video:
- Gender (male/female)
- Approximate age range
- Physical appearance (body type, posture, height estimate)
- Clothing style and colors
- Facial expression and specific details (touching forehead, smiling, frowning, etc.)
- Background/scenery details
- Any objects, furniture, or environment visible
- Body language and positioning
- Any actions or gestures being performed

Only after providing these specific visual details should you proceed to psychological assessment, and ALL psychological conclusions must be supported by the visual evidence you described.

IMPORTANT FORMATTING INSTRUCTION: Do not use ANY markdown formatting in your response. Do not use hashtags (#), asterisks (*), or any other markdown symbols. Use plain text only with clear paragraph breaks and section headings using simple text.

${videoAnalysis ? 'This analysis includes video data showing gestures, activities, and attention patterns.' : ''}
${audioTranscription ? 'This analysis includes audio transcription and speech pattern data.' : ''}

MANDATORY ANALYSIS STRUCTURE:
1. VISUAL DESCRIPTION FIRST: Start with 2-3 paragraphs describing exactly what you see - gender, age, clothes, posture, facial expressions, background, specific details like hand positions, etc.
2. SPEECH/TEXT INTEGRATION: If audio transcription or text data is available, make this the PRIMARY SOURCE for personality analysis. Analyze word choice, communication patterns, topics discussed, emotional tone, and speaking style in extraordinary detail
3. DEEP COGNITIVE PROFILING: Conduct extensive assessment of intellectual capabilities through vocabulary complexity, reasoning patterns in speech, problem-solving approaches mentioned, communication sophistication, and mental agility
4. COMPREHENSIVE PSYCHOLOGICAL PROFILING: Provide detailed analysis of personality traits revealed through what the person says, how they express themselves, their interests, concerns, emotional expressions, values, and worldview
5. EVIDENCE-BASED REASONING: For every assessment, cite multiple specific examples from speech content, direct quotes, and observable visual patterns with detailed explanations
6. EXTENSIVE CONTENT ANALYSIS: Thoroughly discuss the actual topics, ideas, and perspectives shared in speech/text and provide deep insights into what these reveal about character, values, intelligence, emotional landscape, and psychological makeup
7. ENHANCED DEPTH: Go beyond surface-level observations to provide profound insights into the person's psychological landscape, emotional patterns, cognitive style, relationship dynamics, and personal growth potential

YOU MUST ANSWER ALL 50 PSYCHOLOGICAL QUESTIONS - ABSOLUTELY MANDATORY - NO EXCEPTIONS:

STRICT REQUIREMENTS:
- EVERY question below MUST have an explicit, detailed answer
- NO "Not assessed", NO placeholders, NO generic responses
- If visual/audio data is limited, make REASONABLE PSYCHOLOGICAL INFERENCES
- Base answers on: speech patterns, word choice, body language, facial expressions, tone, topics discussed
- Provide SPECIFIC EVIDENCE for each conclusion
- "Not assessed" or similar evasive responses are COMPLETELY UNACCEPTABLE

${videoAnalysis ? `
Answer these 50 VIDEO ANALYSIS questions with substantive, evidence-based responses:

I. Physical & Behavioral Cues (10):
1. How does the person's gait or movement rhythm change across the clip?
2. Which recurring gesture seems habitual rather than situational?
3. Describe one moment where muscle tension releases or spikes — what triggers it?
4. How does posture vary when the person speaks vs. listens?
5. Identify one micro-adjustment (e.g., hair touch, collar fix) and explain its likely emotional cause.
6. What is the person doing with their hands during silent intervals?
7. How consistent is eye-contact across frames? Give timestamps showing breaks or sustained gazes.
8. At which point does breathing rate visibly change, and what precedes it?
9. Describe the physical energy level throughout — rising, falling, or cyclical?
10. What body part seems most expressive (eyes, shoulders, mouth), and how is that used?

II. Expression & Emotion Over Time (10):
11. Track micro-expressions that flicker and vanish. At what timestamps do they appear?
12. When does the dominant emotion shift, and how abruptly?
13. Does the person's smile fade naturally or snap off?
14. Which emotion seems performed vs. spontaneous? Cite frames.
15. How does blink rate change when discussing specific topics?
16. Identify one involuntary facial tic and interpret its significance.
17. Are there moments of incongruence between facial expression and vocal tone?
18. When does the person's face 'freeze' — i.e., hold still unnaturally — and what triggers that?
19. What subtle expression signals discomfort before any verbal cue?
20. How does lighting or camera angle amplify or mute visible emotions?

III. Speech, Voice & Timing (10):
21. Describe baseline vocal timbre — breathy, clipped, resonant — and what personality trait it implies.
22. At which timestamp does pitch spike or flatten dramatically? Why?
23. How does speaking rate change when emotionally charged content arises?
24. Identify one pause longer than 1.5 seconds and interpret it psychologically.
25. What filler words or vocal tics recur, and what function do they serve?
26. How synchronized are gestures with speech rhythm?
27. Does the voice carry underlying fatigue, tension, or confidence? Provide audible markers.
28. Compare early vs. late segments: does articulation become more or less precise?
29. What is the emotional contour of the voice across the clip (anxious → calm, etc.)?
30. When does volume drop below baseline, and what coincides with it visually?

IV. Context, Environment & Interaction (10):
31. What environmental cues (background noise, lighting shifts) change mid-video?
32. How does the camera distance or angle influence perceived dominance or submission?
33. If others are present, how does the subject's behavior shift in their presence vs. alone?
34. What objects does the person interact with, and what do those choices reveal?
35. Is the setting staged or spontaneous, and what does that say about self-presentation?
36. How does background (clean, cluttered, symbolic) reflect personality or mood?
37. Are there off-screen events the person reacts to? What do those reactions reveal?
38. What non-verbal cues suggest comfort or discomfort with being recorded?
39. How does spatial positioning (centered, off-center, close, distant) communicate self-concept?
40. If there's editing or cuts, what moments are emphasized or omitted, and why might that matter?

V. Personality & Psychological Inference (10):
41. What core emotional need appears to drive the person's on-camera behavior?
42. How does the person manage vulnerability — reveal it, suppress it, perform it?
43. What does the rhythm of the video (slow, frenetic, erratic) suggest about internal state?
44. Identify one moment that feels unguarded vs. one that feels calculated. What's the difference?
45. How does the person relate to the camera — as ally, enemy, mirror, or audience?
46. What psychological defense mechanism is most visible (deflection, projection, rationalization)?
47. Does the person seek approval, assert dominance, withdraw, or connect? How do you know?
48. What childhood or formative pattern might explain a recurring behavioral motif?
49. If this person had a recurring dream, what would it be, based on visible themes?
50. What single frame or moment best encapsulates the subject's psychological essence?
` : `
Answer these 50 PHOTO ANALYSIS questions with substantive, evidence-based responses:

I. Physical Cues (10):
1. What is the person's approximate age range, and what visual evidence supports this?
2. What is their likely dominant hand, based on body posture or hand use?
3. What kind of lighting was used (natural, fluorescent, LED), and how does it shape facial tone or mood?
4. How symmetrical is the person's face, and what asymmetries are visible?
5. Describe the color and apparent texture of the person's skin in objective terms.
6. Identify one visible physical trait (scar, mole, wrinkle pattern) and infer its probable significance (age, stress, lifestyle).
7. What can be inferred about the person's sleep habits from the eyes and skin tone?
8. Describe the person's hair (color, grooming, direction, style) and what it indicates about self-presentation.
9. What kind of lighting shadow falls across the eyes or nose, and what mood does that lighting convey?
10. Is there evidence of cosmetic enhancement (makeup, filters, retouching), and how does it alter authenticity?

II. Expression & Emotion (10):
11. Describe the dominant facial expression in granular terms (eyebrow position, lip tension, gaze angle).
12. Does the expression look posed or spontaneous? Why?
13. Identify micro-expressions suggesting secondary emotions (e.g., contempt, anxiety, curiosity).
14. Does the smile (if any) engage the eyes? What does that reveal psychologically?
15. Compare upper-face emotion vs. lower-face emotion; do they match?
16. What emotional tone is conveyed by the person's gaze direction (camera, away, downward)?
17. Does the person appear guarded, open, or performative? Cite visible evidence.
18. Are there tension points in the jaw or neck suggesting repressed emotion?
19. Estimate how long the expression was held for the photo.
20. Does the emotion appear congruent with the setting or mismatched? What does that mismatch suggest?

III. Composition & Context (10):
21. Describe the setting (indoor/outdoor, professional/personal) and how it relates to self-presentation.
22. What objects or background details signal aspects of lifestyle or occupation?
23. How does clothing color palette interact with lighting to create an emotional tone?
24. What focal length or camera distance was likely used, and how does it affect psychological intimacy?
25. Is there visible clutter or minimalism, and what does that suggest about personality?
26. Are there reflections, windows, or mirrors in frame? What might they symbolize?
27. How does body posture interact with spatial framing (e.g., leaning toward/away from camera)?
28. What portion of the frame the subject occupies, and what does that say about ego strength or humility?
29. Does the color grading lean warm, cool, or neutral, and what psychological effect does that create?
30. Is the photo cropped tightly or spaciously? What does that suggest about boundaries or openness?

IV. Personality & Psychological Inference (10):
31. What does the chosen pose (formal, casual, confrontational, withdrawn) reveal about how the person wants to be seen?
32. Is there an attempt to project power, warmth, mystery, or vulnerability? How?
33. Does the person seem comfortable being photographed, or is there visible self-consciousness?
34. What emotional archetype does the image evoke (hero, victim, artist, authority figure)?
35. How does the balance between concealment and revelation define their self-presentation?
36. What does the image omit or hide, and what might that suggest about shame or insecurity?
37. If you had to name one psychological conflict visible in the photo, what would it be?
38. Does the image suggest introversion or extraversion? Support with visual evidence.
39. What story is the person trying to tell about themselves through this image?
40. How does this image relate to social identity (professional, personal, aspirational)?

V. Symbolic & Metapsychological Analysis (10):
41. What does the image say about the subject's relationship with time (past, present, future)?
42. Are there symbolic elements (books, tech, nature) that suggest values or worldview?
43. How does the interplay of light and shadow mirror internal psychological contrasts?
44. What existential theme (freedom, constraint, connection, isolation) does the composition suggest?
45. If this were a still from a film, what genre would it be, and what does that reveal?
46. What emotional temperature (warm, cold, neutral) dominates, and what does it imply about inner life?
47. Does the image feel static or dynamic? What does that suggest about psychological momentum?
48. What is the implied relationship between the photographer and subject (trust, tension, dominance)?
49. If this image were part of a sequence, what emotional narrative would it tell?
50. What single object or feature in the photo best symbolizes the person's life stance?
`}

Return a JSON object with the following structure:
{
  "summary": "Start with detailed visual description (gender, age, clothes, posture, background, specific actions like hand positions) then provide brief psychological overview of ${personLabel}",
  "detailed_analysis": {
    "core_psychological_assessment": {
      "core_motivation": "What drives this person based on speech content, visual cues, and behavioral evidence",
      "confidence_level": "Assessment of their confidence level with supporting evidence from demeanor and speech",
      "self_acceptance": "Do they genuinely like themselves - evidence from behavior and expression",
      "intelligence_level": "Intelligence assessment with evidence from vocabulary, reasoning, and cognitive patterns",
      "creativity_assessment": "Creativity level with evidence from expression, ideas, and approach to topics",
      "stress_handling": "How they handle stress based on demeanor, speech patterns, and behavioral cues",
      "trustworthiness": "Trustworthiness assessment with evidence from consistency and authenticity markers",
      "authenticity": "Authenticity assessment - genuine vs performative elements with evidence",
      "ambition_level": "Ambition level with evidence from goals, drive, and achievement orientation",
      "insecurities": "What they're insecure about - evidence from defensive patterns or overcompensation",
      "social_validation": "Need for external validation vs internal confidence with supporting evidence",
      "independence": "Independent thinking vs conformity with evidence from opinions and choices",
      "communication_style": "Communication approach with evidence from speech patterns and interaction style",
      "criticism_response": "How they likely handle criticism based on defensive patterns and receptiveness",
      "outlook": "Optimistic vs pessimistic tendencies with evidence from expression and content",
      "humor_sense": "Sense of humor assessment with evidence from expression and content",
      "treatment_of_others": "How they treat others with evidence from respect patterns and social awareness",
      "consistency": "Consistency in behavior and thinking with evidence from logical coherence",
      "hidden_strengths": "Hidden positive qualities with subtle evidence from behavior and expression",
      "hidden_weaknesses": "Hidden vulnerabilities with subtle evidence from behavioral patterns"
    },
    "cognitive_profile": {
      "intelligence_assessment": "Estimated intelligence level with specific evidence from speech patterns, vocabulary, problem-solving approaches",
      "cognitive_strengths": ["Specific cognitive abilities that appear well-developed with evidence"],
      "cognitive_weaknesses": ["Areas of cognitive limitation with supporting evidence"],
      "processing_style": "How this person processes information (analytical vs intuitive, sequential vs random, etc.) with evidence",
      "mental_agility": "Assessment of mental flexibility and adaptability with examples"
    },
    "personality_core": "Deep analysis of core personality traits with specific evidence from facial expressions, body language, speech patterns",
    "thought_patterns": "Analysis of cognitive processes and decision-making style with supporting evidence",
    "emotional_intelligence": "Assessment of emotional awareness and social intelligence with observable evidence",
    "behavioral_indicators": "Specific behaviors observed that reveal personality traits",
    "speech_analysis": {
      "key_quotes": ["Include 5-8 direct quotes from the transcription that reveal personality traits, interests, values, and thinking patterns"],
      "content_themes": "Detailed analysis of the main topics, ideas, and subjects the person discusses and what these reveal about their interests, expertise, and priorities",
      "vocabulary_analysis": "Analysis of word choice, complexity, sophistication level, and communication patterns with specific examples",
      "speech_patterns": "Analysis of speech patterns, pace, tone, communication style, and conversational approach",
      "emotional_tone": "Analysis of emotional tone, enthusiasm, concerns, and feelings expressed in speech with specific examples",
      "personality_revealed": "What the actual content of their speech reveals about their character, values, beliefs, and worldview with direct evidence"
    },
    "visual_evidence": {
      "physical_description": "Detailed description of gender, approximate age, body type, clothing, and overall appearance",
      "facial_expressions": "Specific facial expressions observed (touching forehead, smiling, frowning, etc.) and what they reveal about personality",
      "body_language": "Specific posture, gestures, hand positions, and physical presence with psychological analysis",
      "environment": "Description of background, scenery, objects, and setting with psychological implications",
      "emotional_indicators": "Observable emotional states and their psychological implications based on specific visual cues"
    },
    "professional_insights": "Career inclinations and work style based on cognitive profile and personality traits",
    "relationships": {
      "current_status": "Likely relationship status based on evidence",
      "parental_status": "Insights about parenting style or potential",
      "ideal_partner": "Description of compatible partner characteristics"
    },
    "growth_areas": {
      "strengths": ["List of key strengths with evidence"],
      "challenges": ["Areas for improvement with specific indicators"],
      "development_path": "Suggested personal growth direction based on cognitive and personality profile"
    }
  }
}

Be EXTRAORDINARILY thorough and insightful. Each section must be 8-12 paragraphs long with EXTENSIVE detail, PROFOUND insights, and COMPREHENSIVE evidence. This is a $10,000 premium analysis - it must be exceptionally detailed and extensive.

ABSOLUTE REQUIREMENTS FOR EXTREME DEPTH - FAILURE TO MEET THESE STANDARDS IS UNACCEPTABLE:
1. EXTENSIVE SPEECH-FIRST ANALYSIS: When transcription/text is available, conduct EXHAUSTIVE analysis of every nuance, word choice, speech pattern, emotional tone, cognitive sophistication, and psychological revelation
2. COMPREHENSIVE CONTENT INTEGRATION: Provide EXTENSIVE discussion of what the person talks about, their interests, concerns, opinions, values, beliefs, fears, aspirations, and how they express themselves with EXTRAORDINARY detail
3. MASSIVE DIRECT QUOTES INTEGRATION: Include 15-20 meaningful quotes that showcase personality, intelligence, values, communication style, thought processes, emotional patterns, and psychological depth
4. PROFOUND COGNITIVE EVIDENCE: Conduct EXTENSIVE analysis of intelligence through vocabulary, reasoning patterns, problem-solving approaches, mental agility, cognitive style, and intellectual sophistication
5. EXTRAORDINARY CHARACTER INSIGHTS: Provide DEEP analysis of what their choice of topics, perspectives, expressions, concerns, and interests reveal about their deeper character, values, worldview, psychological makeup, and life philosophy
6. COMPREHENSIVE VISUAL ANALYSIS: Reference specific facial expressions, micro-expressions, body language, posture, gestures, and visual cues with THOROUGH psychological interpretation
7. PREMIUM PROFESSIONAL EXCELLENCE: Maintain scientific objectivity while providing EXTRAORDINARILY detailed, actionable insights that demonstrate exceptional psychological expertise
8. MAXIMUM PSYCHOLOGICAL DEPTH: Provide EXTENSIVE insights into emotional patterns, defense mechanisms, attachment styles, communication strategies, relationship dynamics, psychological vulnerabilities, strengths, and personal growth potential
9. EXTENSIVE EVIDENCE INTEGRATION: Every single assessment must be supported by MULTIPLE specific examples, quotes, observations, and behavioral indicators
10. COMPREHENSIVE LIFE ANALYSIS: Analyze career potential, relationship compatibility, parenting style, leadership qualities, emotional intelligence, social dynamics, and personal development needs in EXTRAORDINARY detail`;

        // Use OpenAI as primary source for consistency across multiple analyses
        try {
          if (!openai) {
            throw new Error("OpenAI client not available");
          }
          
          const response = await openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
              {
                role: "system",
                content: analysisPrompt,
              },
              {
                role: "user",
                content: JSON.stringify(personInput),
              },
            ],
            response_format: { type: "json_object" },
          });
          
          // Parse and validate results
          const analysisResult = JSON.parse(response.choices[0]?.message.content || "{}");
          
          // Validate that all 20 core questions are answered
          validateCoreAssessment(analysisResult, personFaceData.personLabel);
          
          return {
            ...analysisResult,
            personLabel: personFaceData.personLabel,
            personIndex: personFaceData.positionInImage,
            // Add positional data for potential UI highlighting
            boundingBox: personFaceData.boundingBox
          };
        } catch (err) {
          console.error(`Failed to analyze ${personLabel}:`, err);
          // Return minimal profile on error
          return {
            summary: `Analysis of ${personLabel} could not be completed.`,
            detailed_analysis: {
              personality_core: "Analysis unavailable for this individual.",
              thought_patterns: "Analysis unavailable.",
              cognitive_style: "Analysis unavailable.",
              professional_insights: "Analysis unavailable.",
              relationships: {
                current_status: "Analysis unavailable.",
                parental_status: "Analysis unavailable.",
                ideal_partner: "Analysis unavailable."
              },
              growth_areas: {
                strengths: ["Unknown"],
                challenges: ["Unknown"],
                development_path: "Analysis unavailable."
              }
            },
            personLabel: personFaceData.personLabel,
            personIndex: personFaceData.positionInImage
          };
        }
      } catch (error) {
        console.error("Error analyzing person:", error);
        return null;
      }
    });
    
    // Wait for all analyses to complete
    const individualResults = await Promise.all(analysisPromises);
    
    // Filter out any failed analyses
    multiPersonAnalysis.individualProfiles = individualResults.filter(result => result !== null);
    
    // Generate a group dynamics summary if we have multiple successful analyses
    if (multiPersonAnalysis.individualProfiles.length > 1) {
      try {
        // Create a combined input with only successful profiles
        const groupInput = {
          profiles: multiPersonAnalysis.individualProfiles.map(profile => ({
            personLabel: profile.personLabel,
            summary: profile.summary,
            key_traits: profile.detailed_analysis.personality_core.substring(0, 200) // Truncate for brevity
          }))
        };
        
        const groupPrompt = `
You are analyzing the group dynamics of ${multiPersonAnalysis.individualProfiles.length} people detected in the same media.
Based on the individual summaries provided, generate a brief analysis of how these personalities might interact.

Return a short paragraph (3-5 sentences) describing potential group dynamics, 
compatibilities or conflicts, and how these different personalities might complement each other.`;

        if (!openai) {
          throw new Error("OpenAI client not available for group dynamics analysis");
        }
        
        const groupResponse = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [
            {
              role: "system",
              content: groupPrompt,
            },
            {
              role: "user",
              content: JSON.stringify(groupInput),
            },
          ]
        });
        
        multiPersonAnalysis.groupDynamics = groupResponse.choices[0]?.message.content || 
          "Group dynamics analysis unavailable.";
      } catch (err) {
        console.error("Error generating group dynamics:", err);
        multiPersonAnalysis.groupDynamics = "Group dynamics analysis unavailable.";
      }
    }
    
    return multiPersonAnalysis;
  } else {
    // Original single-person analysis logic
    // Build a comprehensive analysis input combining all the data we have
    const analysisInput = {
      faceAnalysis,
      ...(videoAnalysis && { videoAnalysis }),
      ...(audioTranscription && { audioTranscription })
    };
    
    // Use PHOTO or VIDEO questions depending on whether we have video/audio
    const questions = (videoAnalysis || audioTranscription) ? VIDEO_ANALYSIS_QUESTIONS : PHOTO_ANALYSIS_QUESTIONS;
    const questionCount = questions.length;
    const mediaType = (videoAnalysis || audioTranscription) ? "VIDEO" : "PHOTO";
    
    const analysisPrompt = `
You are the world's most elite psychological profiler conducting a comprehensive forensic-level ${mediaType} personality assessment. You must answer ALL ${questionCount} QUESTIONS with specific visual and/or audio evidence.

CRITICAL REQUIREMENT: YOU MUST START WITH DETAILED VISUAL DESCRIPTION. Begin your analysis by describing what you actually see in the image/video:
- Gender (male/female)
- Approximate age range
- Physical appearance (body type, posture, height estimate, weight)
- Clothing style, colors, and formality level
- Facial expression and specific details (touching forehead, smiling, frowning, hand positions, etc.)
- Background/scenery details (office, home, outdoor, furniture, objects)
- Any objects, technology, or personal items visible
- Body language and positioning
- Any actions or gestures being performed
- Grooming and styling choices

Only after providing these specific visual details should you proceed to psychological assessment, and ALL psychological conclusions must be supported by the visual evidence you described.

CRITICAL REQUIREMENTS:
- NO MARKDOWN FORMATTING: Do not use # ### ** or any markdown
- PROVIDE SPECIFIC EVIDENCE for each answer:
  * VISUAL EVIDENCE: appearance details, body language, posture, facial expressions, clothing, grooming, background objects, weight/build, microexpressions
  * AUDIO EVIDENCE: direct quotations from speech (if available), tone of voice, speaking patterns
- Answer each question with 2-3 sentences of detailed analysis and supporting evidence

MANDATORY ${questionCount} PSYCHOLOGICAL QUESTIONS - ANSWER ALL WITH SPECIFIC EVIDENCE:

${questions.map((q, i) => `${i + 1}. ${q} - Provide specific visual/audio evidence for your assessment`).join('\n')}



V. SELF & EGO STRUCTURE
- How integrated or fragmented does the self appear?
- Are there defenses (denial, projection, reaction formation, intellectualization)? Identify with examples.
- Is the ego brittle, grandiose, or well-regulated?
- How does the subject relate to authority or ideals? Quote if available.
- Where are points of narcissistic injury or vulnerability visible?

VI. INTERPERSONAL STYLE
- How does the subject implicitly treat the viewer/listener? Quote if available.
- Is there a manipulative undertone — flattery, intimidation, seduction, deflection?
- What role would they likely assume in a group (leader, scapegoat, clown, father)?
- Do they show capacity for empathy or only self-reference? Quote if available.
- Are intimacy and distance well-calibrated or distorted?

VII. CULTURAL & SYMBOLIC POSITION
- What class, cultural, or ideological identity is signaled?
- Are there signs of ressentiment, envy, or superiority?
- Does the subject echo stock phrases or clichés? Quote if available.
- Are there unconscious cultural archetypes being embodied (hero, victim, trickster)?
- What symbolic resonance does the subject's presentation carry (mythic, banal, tragic)?

VIII. PROGNOSIS & DYNAMICS
- What long-term psychological trajectory does this person's style suggest?
- What specific pathologies or neuroses are hinted at (obsessional, hysteric, narcissistic)?
- What are latent strengths — creativity, resilience, discipline, insight?
- What situational stressors would break down this persona?
- How might this individual evolve if confronted with loss, failure, or aging?

CRITICAL ANALYSIS REQUIREMENTS - FORENSIC-LEVEL DEPTH:

MICRO-DETAIL ANALYSIS: Identify specific micro-expressions, asymmetries, posture details, clothing choices, and unconscious signals with forensic precision (e.g., "Unilateral lip pull = contempt/irony; narrowed eyes = skeptical focus; forward chin tilt = defiance")

MULTI-SOURCE DATA INTEGRATION: This analysis integrates comprehensive data from ALL available services:
${videoAnalysis ? '- Azure Video Indexer deep behavioral analysis with temporal emotion tracking' : ''}
${audioTranscription ? '- Multi-provider audio transcription with sentiment analysis and entity detection' : ''}
- Face++ comprehensive facial analysis (emotions, ethnicity, beauty scores, micro-expressions)
- Azure Face API detailed emotional and physical attribute detection
- Google Cloud Vision facial landmarks and object detection
- AWS Rekognition multi-person facial feature analysis
- Cross-service data validation and psychological indicator correlation

FORENSIC DEPTH REQUIREMENTS:
- 8-12 paragraphs per major framework section with exhaustive detail
- 15-20 meaningful direct quotes from speech/text when available
- Micro-signal analysis with specific facial measurements and asymmetries
- Cross-reference multiple AI service outputs for validation
- Answer EVERY framework question with comprehensive evidence
- Pure psychological passthrough - no content filtering or sanitization

Return a JSON object with this structure - YOU MUST ANSWER ALL 20 CORE QUESTIONS WITH EVIDENCE:
{
  "summary": "Start with detailed visual description (male/female, age, body type, clothes, posture, hand positions, background, etc.) then provide comprehensive 2-3 paragraph personality overview",
  "detailed_analysis": {
    "core_psychological_assessment": {
      "core_motivation": "What drives this person - provide specific visual evidence (posture, expression, clothing) and quotes if available",
      "confidence_level": "How confident are they - analyze body language, eye contact, posture, facial expression with specific details",
      "self_acceptance": "Do they genuinely like themselves - evidence from facial expression, grooming care, posture, self-referential speech",
      "intelligence_level": "How smart are they - assess through facial alertness, eye engagement, speech complexity, background objects",
      "creativity_assessment": "How creative are they - evidence from clothing choices, environment, unique expressions, original speech",
      "stress_handling": "How they handle stress - body tension, facial strain, defensive postures, stress-related speech patterns",
      "trustworthiness": "Are they trustworthy - eye contact quality, facial openness, genuine vs forced expressions, speech consistency",
      "authenticity": "Do they exaggerate or fake things - performative vs natural expressions, posed vs candid appearance",
      "ambition_level": "How ambitious are they - assertive posture, determined expression, professional presentation, goal-oriented speech",
      "insecurities": "What are they insecure about - defensive body language, self-conscious expressions, covering behaviors, hesitant speech",
      "social_validation": "How much do they care what others think - posed vs natural appearance, grooming attention, performative expressions",
      "independence": "Are they independent-minded or followers - unique vs conventional appearance, original expressions, unconventional speech",
      "communication_style": "Do they dominate or listen more - assertive vs receptive body language, eye contact patterns, speech volume",
      "criticism_response": "How do they deal with criticism - defensive postures, facial reactions, openness vs closed-off body language",
      "outlook": "Are they optimistic or pessimistic - facial expression positivity, upward vs downward body language, speech tone",
      "humor_sense": "Do they have strong sense of humor - eye crinkles, smile genuineness, playful expressions, humorous speech",
      "treatment_of_others": "How do they treat people beneath them - facial warmth vs coldness, respectful vs dismissive posture",
      "consistency": "Are they consistent or contradictory - expression authenticity, body language alignment, speech consistency",
      "hidden_strengths": "What hidden strengths do they have - subtle confident details, understated competence signals, quiet strength",
      "hidden_weaknesses": "What hidden weaknesses do they have - subtle tension signs, compensatory behaviors, masked insecurities"
    },
    "speech_analysis": {
      "key_quotes": ["meaningful quotes from speech that reveal personality traits", "quote showing intelligence", "quote revealing values", "quote demonstrating communication style"],
      "vocabulary_analysis": "analysis of word choice, linguistic sophistication, communication style with specific examples",
      "personality_revealed": "detailed insights into character traits revealed through specific speech patterns and word choices"
    },
    "visual_evidence": {
      "facial_analysis": "detailed analysis of facial expressions, microexpressions, eye contact, smile authenticity with specific observations",
      "body_language": "comprehensive analysis of posture, gestures, defensive vs open positioning with specific details", 
      "appearance_details": "clothing choices, grooming, weight/build, background objects and what they reveal about personality",
      "microexpressions": "specific micro-expressions observed and their psychological significance"
    },
    "professional_insights": "comprehensive analysis of career inclinations, work style preferences, leadership qualities based on visual and audio evidence",
    "growth_areas": {
      "strengths": ["strength 1 with detailed visual/audio evidence", "strength 2 with evidence", "strength 3 with evidence"],
      "development_path": "detailed recommendations for personal and professional growth based on observed patterns and evidence"
    }
  }
}

Be EXTRAORDINARILY thorough and insightful. Each section must be 8-12 paragraphs long with EXTENSIVE detail, PROFOUND insights, and COMPREHENSIVE evidence. This is a $10,000 premium analysis - it must be exceptionally detailed and extensive.

ABSOLUTE REQUIREMENTS FOR EXTREME DEPTH - FAILURE TO MEET THESE STANDARDS IS UNACCEPTABLE:
1. EXTENSIVE SPEECH-FIRST ANALYSIS: When transcription/text is available, conduct EXHAUSTIVE analysis of every nuance, word choice, speech pattern, emotional tone, cognitive sophistication, and psychological revelation
2. COMPREHENSIVE CONTENT INTEGRATION: Provide EXTENSIVE discussion of what the person talks about, their interests, concerns, opinions, values, beliefs, fears, aspirations, and how they express themselves with EXTRAORDINARY detail
3. MASSIVE DIRECT QUOTES INTEGRATION: Include 15-20 meaningful quotes that showcase personality, intelligence, values, communication style, thought processes, emotional patterns, and psychological depth
4. PROFOUND COGNITIVE EVIDENCE: Conduct EXTENSIVE analysis of intelligence through vocabulary, reasoning patterns, problem-solving approaches, mental agility, cognitive style, and intellectual sophistication
5. EXTRAORDINARY CHARACTER INSIGHTS: Provide DEEP analysis of what their choice of topics, perspectives, expressions, concerns, and interests reveal about their deeper character, values, worldview, psychological makeup, and life philosophy
6. COMPREHENSIVE VISUAL ANALYSIS: Reference specific facial expressions, micro-expressions, body language, posture, gestures, and visual cues with THOROUGH psychological interpretation
7. PREMIUM PROFESSIONAL EXCELLENCE: Maintain scientific objectivity while providing EXTRAORDINARILY detailed, actionable insights that demonstrate exceptional psychological expertise
8. MAXIMUM PSYCHOLOGICAL DEPTH: Provide EXTENSIVE insights into emotional patterns, defense mechanisms, attachment styles, communication strategies, relationship dynamics, psychological vulnerabilities, strengths, and personal growth potential
9. EXTENSIVE EVIDENCE INTEGRATION: Every single assessment must be supported by MULTIPLE specific examples, quotes, observations, and behavioral indicators
10. COMPREHENSIVE LIFE ANALYSIS: Analyze career potential, relationship compatibility, parenting style, leadership qualities, emotional intelligence, social dynamics, and personal development needs in EXTRAORDINARY detail`;

    // Try to get analysis from all three services in parallel for maximum depth
    try {
      // Prepare API calls based on available clients
      const apiPromises = [];
      
      // OpenAI Analysis (if available)
      if (openai) {
        apiPromises.push(
          openai.chat.completions.create({
            model: "gpt-4o",
            messages: [
              {
                role: "system",
                content: analysisPrompt,
              },
              {
                role: "user",
                content: JSON.stringify(analysisInput),
              },
            ],
            response_format: { type: "json_object" },
          })
        );
      } else {
        apiPromises.push(Promise.reject(new Error("OpenAI client not available")));
      }
      
      // Anthropic Analysis (if available)
      if (anthropic) {
        apiPromises.push(
          anthropic.messages.create({
            model: "claude-sonnet-4-20250514", // the newest Anthropic model is "claude-sonnet-4-20250514", not "claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022" nor "claude-3-sonnet-20240229"
            max_tokens: 8000,
            system: analysisPrompt,
            messages: [
              {
                role: "user",
                content: JSON.stringify(analysisInput),
              }
            ],
          })
        );
      } else {
        apiPromises.push(Promise.reject(new Error("Anthropic client not available")));
      }
      
      // Perplexity Analysis (if API key available)
      if (process.env.PERPLEXITY_API_KEY) {
        apiPromises.push(
          perplexity.query({
            model: "mistral-large-latest",
            query: `${analysisPrompt}\n\nHere is the data to analyze: ${JSON.stringify(analysisInput)}`,
          })
        );
      } else {
        apiPromises.push(Promise.reject(new Error("Perplexity API key not available")));
      }
      
      // Run all API calls in parallel
      const [openaiResult, anthropicResult, perplexityResult] = await Promise.allSettled(apiPromises);
      
      // Process results from each service
      let finalInsights: any = {};
      
      // Try each service result in order of preference (Anthropic first as specified in requirements)
      if (anthropicResult.status === 'fulfilled') {
        try {
          // Handle Anthropic API response structure
          const anthropicResponse = anthropicResult.value as any;
          if (anthropicResponse.content && Array.isArray(anthropicResponse.content) && anthropicResponse.content.length > 0) {
            const content = anthropicResponse.content[0];
            const anthropicData = JSON.parse(content.text || "{}");
            finalInsights = anthropicData;
            console.log("Anthropic Claude analysis used as primary source");
          }
        } catch (e) {
          console.error("Error parsing Anthropic response:", e);
        }
      } else if (openaiResult.status === 'fulfilled') {
        try {
          // Handle OpenAI response
          const openaiResponse = openaiResult.value as any;
          const openaiData = JSON.parse(openaiResponse.choices[0]?.message.content || "{}");
          finalInsights = openaiData;
          console.log("OpenAI analysis used as secondary source");
        } catch (e) {
          console.error("Error parsing OpenAI response:", e);
        }
      } else if (perplexityResult.status === 'fulfilled') {
        try {
          // Handle Anthropic API response structure
          const anthropicResponse = anthropicResult.value as any;
          if (anthropicResponse.content && Array.isArray(anthropicResponse.content) && anthropicResponse.content.length > 0) {
            const content = anthropicResponse.content[0];
            // Check if it's a text content type
            if (content && content.type === 'text') {
              const anthropicText = content.text;
              // Extract JSON from Anthropic response (which might include markdown formatting)
              const jsonMatch = anthropicText.match(/```json\n([\s\S]*?)\n```/) || 
                                anthropicText.match(/{[\s\S]*}/);
                                
              if (jsonMatch) {
                const jsonStr = jsonMatch[1] || jsonMatch[0];
                finalInsights = JSON.parse(jsonStr);
                console.log("Anthropic analysis used as backup");
              }
            }
          }
        } catch (e) {
          console.error("Error parsing Anthropic response:", e);
        }
      } else if (perplexityResult.status === 'fulfilled') {
        try {
          // Extract JSON from Perplexity response
          const perplexityResponse = perplexityResult.value as any;
          const perplexityText = perplexityResponse.text || "";
          const jsonMatch = perplexityText.match(/```json\n([\s\S]*?)\n```/) || 
                           perplexityText.match(/{[\s\S]*}/);
                           
          if (jsonMatch) {
            const jsonStr = jsonMatch[1] || jsonMatch[0];
            finalInsights = JSON.parse(jsonStr);
            console.log("Perplexity analysis used as backup");
          }
        } catch (e) {
          console.error("Error parsing Perplexity response:", e);
        }
      }
      
      // If we couldn't get analysis from any service, fall back to a basic structure
      if (!finalInsights || Object.keys(finalInsights).length === 0) {
        console.error("All personality analysis services failed, using basic fallback");
        finalInsights = {
          summary: "Analysis could not be completed fully.",
          detailed_analysis: {
            personality_core: "The analysis could not be completed at this time. Please try again with a clearer image or video.",
            thought_patterns: "Analysis unavailable.",
            cognitive_style: "Analysis unavailable.",
            professional_insights: "Analysis unavailable.",
            relationships: {
              current_status: "Analysis unavailable.",
              parental_status: "Analysis unavailable.",
              ideal_partner: "Analysis unavailable."
            },
            growth_areas: {
              strengths: ["Determination"],
              challenges: ["Technical issues"],
              development_path: "Try again with a clearer image or video."
            }
          }
        };
      }
      
      // VALIDATE: Ensure all 20 core psychological questions are answered
      validateCoreAssessment(finalInsights, "Subject");
      
      // Enhance with combined insights if we have multiple services working
      if (openaiResult.status === 'fulfilled' && (anthropicResult.status === 'fulfilled' || perplexityResult.status === 'fulfilled')) {
        finalInsights.provider_info = "This analysis used multiple AI providers for maximum depth and accuracy.";
      }
      
      // For single person case, wrap in object with peopleCount=1 for consistency
      return {
        peopleCount: 1,
        individualProfiles: [finalInsights],
        detailed_analysis: finalInsights.detailed_analysis || {} // For backward compatibility
      };
    } catch (error) {
      console.error("Error in getPersonalityInsights:", error);
      throw new Error("Failed to generate personality insights. Please try again.");
    }
  }
}