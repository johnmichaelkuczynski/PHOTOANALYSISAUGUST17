import mail from '@sendgrid/mail';
import { Share, Analysis } from '../../shared/schema';

// Check if required environment variables are set
if (!process.env.SENDGRID_API_KEY) {
  console.warn("Warning: SENDGRID_API_KEY environment variable is not set. Email functionality will be disabled.");
} else {
  // Initialize the SendGrid client with the API key
  mail.setApiKey(process.env.SENDGRID_API_KEY);
  console.log("SendGrid client initialized with API key");
}

if (!process.env.SENDGRID_VERIFIED_SENDER) {
  console.warn("Warning: SENDGRID_VERIFIED_SENDER environment variable is not set. Using default sender.");
}

// Set verified sender email or use a default (which won't work without verification)
const FROM_EMAIL = process.env.SENDGRID_VERIFIED_SENDER || 'notifications@personality-insights.app';
console.log(`SendGrid configured with sender email: ${FROM_EMAIL}`);

interface SendAnalysisEmailParams {
  share: Share;
  analysis: Analysis;
  shareUrl?: string;
}

export async function sendAnalysisEmail({
  share,
  analysis,
  shareUrl
}: SendAnalysisEmailParams): Promise<boolean> {
  try {
    // If SendGrid is not configured, log error and return false
    if (!process.env.SENDGRID_API_KEY) {
      console.error('SendGrid is not configured. Emails cannot be sent.');
      return false;
    }
    
    console.log(`[SendGrid] Attempting to send email from ${FROM_EMAIL} to ${share.recipientEmail}`);
    console.log(`[SendGrid] Using analysis ID: ${analysis.id}, share ID: ${share.id}`);
    console.log(`[SendGrid] Share URL: ${shareUrl}`);
    

    // Parse the JSON data from the database
    const personalityInsights = analysis.personalityInsights as any || {};
    
    // Create a type-safe object to ensure we can safely access properties
    const typeSafeInsights = {
      individualProfiles: personalityInsights.individualProfiles || [],
      peopleCount: personalityInsights.peopleCount || 1,
      summary: personalityInsights.summary || 'No summary available',
      personality_core: personalityInsights.personality_core || {},
      thought_patterns: personalityInsights.thought_patterns || {},
      professional_insights: personalityInsights.professional_insights || {},
      growth_areas: personalityInsights.growth_areas || {},
    };
    
    // Detect if we have a multi-person analysis
    const isMultiPersonAnalysis = typeSafeInsights.individualProfiles.length > 1;
    
    // Get total people count
    const peopleCount = typeSafeInsights.peopleCount;
    
    // For backward compatibility with single-person analysis
    let summary = typeSafeInsights.summary;
    let detailedAnalysis: any = {
      personality_core: '',
      thought_patterns: '',
      professional_insights: '',
      growth_areas: {
        strengths: [],
        challenges: [],
        development_path: ''
      }
    };

    // Determine if this was a video analysis that includes transcription
    const isVideoAnalysis = analysis.mediaType === 'video';
    
    // Get transcription data - it could be stored in multiple places based on our implementation
    // First, check if it's stored directly in audioTranscription field
    let transcription = '';
    if (analysis.audioTranscription) {
      const audioData = analysis.audioTranscription as any;
      transcription = audioData.transcription || '';
    } 
    // If not found, check if it's nested in videoAnalysis
    else if (analysis.videoAnalysis) {
      const videoData = analysis.videoAnalysis as any;
      if (videoData.audioTranscription) {
        transcription = videoData.audioTranscription.transcription || '';
      }
    }
    
    // Format content differently based on single vs. multiple people
    let emailContent = '';
    
    if (isMultiPersonAnalysis) {
      // Handle multi-person analysis
      const profiles = personalityInsights.individualProfiles || [];
      const overviewSummary = personalityInsights.overviewSummary || `Analysis of ${peopleCount} people detected in the media.`;
      
      // Generate group analysis HTML
      emailContent = `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #333;">Your Multi-Person Analysis Results</h2>
          
          <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3 style="color: #444;">Overview (${peopleCount} People Detected)</h3>
            <p>${overviewSummary}</p>
          </div>
          
          ${isVideoAnalysis && transcription ? `
          <div style="background: #f5f5ff; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #6366f1;">
            <h3 style="color: #4338ca;">Video Speech Transcription</h3>
            <p style="font-style: italic;">"${transcription}"</p>
          </div>
          ` : ''}
          
          ${personalityInsights.groupDynamics ? `
          <div style="background: #f0fff4; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #10b981;">
            <h3 style="color: #059669;">Group Dynamics</h3>
            <p>${personalityInsights.groupDynamics}</p>
          </div>
          ` : ''}
      `;
      
      // Add individual profiles section
      emailContent += `
        <h3 style="color: #333; margin-top: 30px; border-bottom: 1px solid #ddd; padding-bottom: 10px;">Individual Profiles</h3>
      `;
      
      // Add each person's profile
      profiles.forEach((profile: any, index: number) => {
        const personLabel = profile.personLabel || `Person ${index + 1}`;
        const personSummary = profile.summary || 'No summary available';
        const personDetails = profile.detailed_analysis || {};
        
        emailContent += `
          <div style="margin: 30px 0; background: #fff; border: 1px solid #eee; border-radius: 10px; padding: 15px; box-shadow: 0 2px 5px rgba(0,0,0,0.05);">
            <h3 style="color: #3b82f6; margin-top: 0;">${personLabel}</h3>
            
            <div style="margin: 15px 0;">
              <h4 style="color: #4b5563; margin-bottom: 5px;">Summary</h4>
              <p>${personSummary}</p>
            </div>
            
            <div style="margin: 15px 0;">
              <h4 style="color: #4b5563; margin-bottom: 5px;">Core Personality</h4>
              <p>${personDetails.personality_core || 'Not available'}</p>
            </div>
            
            <div style="margin: 15px 0;">
              <h4 style="color: #4b5563; margin-bottom: 5px;">Professional Insights</h4>
              <p>${personDetails.professional_insights || 'Not available'}</p>
            </div>
            
            <div style="margin: 15px 0;">
              <h4 style="color: #4b5563; margin-bottom: 5px;">Key Strengths</h4>
              <ul style="margin-top: 5px;">
                ${Array.isArray(personDetails.growth_areas?.strengths) 
                  ? personDetails.growth_areas.strengths.map((s: string) => `<li>${s}</li>`).join('') 
                  : '<li>Not available</li>'}
              </ul>
            </div>
          </div>
        `;
      });
      
      // Add footer
      emailContent += `
        <div style="background: #eef4ff; padding: 15px; border-radius: 5px; margin: 20px 0;">
          <p style="font-style: italic;">This multi-person analysis was shared by: ${share.senderEmail}</p>
          <p style="font-size: 0.8em; color: #666; margin-top: 10px;">Analysis based on ${isVideoAnalysis ? 'video' : 'image'} processing with advanced AI technology.</p>
          ${shareUrl ? `<p style="margin-top: 15px;"><a href="${shareUrl}" style="display: inline-block; background: #3b82f6; color: white; padding: 8px 15px; text-decoration: none; border-radius: 4px; font-weight: 500;">View Analysis Online</a></p>` : ''}
        </div>
      </div>
      `;
      
    } else {
      // Get data for single-person format (traditional format)
      if (typeSafeInsights.individualProfiles?.length === 1) {
        const profile = typeSafeInsights.individualProfiles[0];
        summary = profile.summary || 'No summary available';
        // Ensure detailedAnalysis has the expected structure
        const profileAnalysis = profile.detailed_analysis || {};
        detailedAnalysis = {
          personality_core: profileAnalysis.personality_core || '',
          thought_patterns: profileAnalysis.thought_patterns || '',
          professional_insights: profileAnalysis.professional_insights || '',
          growth_areas: {
            strengths: Array.isArray(profileAnalysis.growth_areas?.strengths) ? profileAnalysis.growth_areas.strengths : [],
            challenges: Array.isArray(profileAnalysis.growth_areas?.challenges) ? profileAnalysis.growth_areas.challenges : [],
            development_path: profileAnalysis.growth_areas?.development_path || ''
          }
        };
      } else {
        // Legacy format
        summary = typeSafeInsights.summary;
        // Ensure detailedAnalysis has the expected structure using the typeSafeInsights
        const legacyAnalysis = personalityInsights.detailed_analysis || {};
        detailedAnalysis = {
          personality_core: legacyAnalysis.personality_core || '',
          thought_patterns: legacyAnalysis.thought_patterns || '',
          professional_insights: legacyAnalysis.professional_insights || '',
          growth_areas: {
            strengths: Array.isArray(legacyAnalysis.growth_areas?.strengths) ? legacyAnalysis.growth_areas.strengths : [],
            challenges: Array.isArray(legacyAnalysis.growth_areas?.challenges) ? legacyAnalysis.growth_areas.challenges : [],
            development_path: legacyAnalysis.growth_areas?.development_path || ''
          }
        };
      }
      
      // Generate single-person email content (original format)
      emailContent = `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
          <h2 style="color: #333;">Your Personality Analysis Results</h2>
          
          <div style="background: #f9f9f9; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <h3 style="color: #444;">Summary</h3>
            <p>${summary}</p>
          </div>

          ${isVideoAnalysis && transcription ? `
          <div style="background: #f5f5ff; padding: 15px; border-radius: 5px; margin: 20px 0; border-left: 4px solid #6366f1;">
            <h3 style="color: #4338ca;">Video Speech Transcription</h3>
            <p style="font-style: italic;">"${transcription}"</p>
          </div>
          ` : ''}

          <div style="margin: 20px 0;">
            <h3 style="color: #444;">Core Personality Traits</h3>
            <p>${detailedAnalysis.personality_core || 'Not available'}</p>
          </div>

          <div style="margin: 20px 0;">
            <h3 style="color: #444;">Thought Patterns</h3>
            <p>${detailedAnalysis.thought_patterns || 'Not available'}</p>
          </div>

          <div style="margin: 20px 0;">
            <h3 style="color: #444;">Professional Insights</h3>
            <p>${detailedAnalysis.professional_insights || 'Not available'}</p>
          </div>

          <div style="margin: 20px 0;">
            <h3 style="color: #444;">Growth Areas</h3>
            <h4 style="color: #666;">Strengths</h4>
            <ul>
              ${(detailedAnalysis.growth_areas?.strengths || []).map((s: string) => `<li>${s}</li>`).join('')}
            </ul>
            
            <h4 style="color: #666;">Challenges</h4>
            <ul>
              ${(detailedAnalysis.growth_areas?.challenges || []).map((c: string) => `<li>${c}</li>`).join('')}
            </ul>
            
            <h4 style="color: #666;">Development Path</h4>
            <p>${detailedAnalysis.growth_areas?.development_path || 'Not available'}</p>
          </div>

          <div style="background: #eef4ff; padding: 15px; border-radius: 5px; margin: 20px 0;">
            <p style="font-style: italic;">This analysis was shared by: ${share.senderEmail}</p>
            <p style="font-size: 0.8em; color: #666; margin-top: 10px;">Analysis based on ${isVideoAnalysis ? 'video' : 'image'} processing with AI technology.</p>
            ${shareUrl ? `<p style="margin-top: 15px;"><a href="${shareUrl}" style="display: inline-block; background: #3b82f6; color: white; padding: 8px 15px; text-decoration: none; border-radius: 4px; font-weight: 500;">View Analysis Online</a></p>` : ''}
          </div>
        </div>
      `;
    }

    // Prepare email data object
    const msg = {
      to: share.recipientEmail,
      from: {
        email: FROM_EMAIL,
        name: "Personality Insights"
      },
      subject: 'Your Personality Analysis Results',
      html: emailContent,
    };
    
    console.log(`[SendGrid] Preparing to send email with subject: ${msg.subject}`);
    console.log(`[SendGrid] From: ${msg.from.name} <${msg.from.email}>`);
    console.log(`[SendGrid] To: ${msg.to}`);
    
    try {
      console.log('[SendGrid] Attempting to send email via SendGrid API...');
      
      // Verify SendGrid is properly configured before sending
      if (!process.env.SENDGRID_API_KEY) {
        throw new Error('SendGrid API key is missing');
      }
      
      if (!process.env.SENDGRID_VERIFIED_SENDER) {
        console.warn('[SendGrid] Warning: Using unverified sender email, email may not be delivered');
      }
      
      const response = await mail.send(msg);
      console.log('[SendGrid] Email sent successfully!', response);
      return true;
    } catch (error) {
      console.error('[SendGrid] Error while sending email:', error);
      // Type assertion for SendGrid error response type
      const sendGridError = error as any;
      if (sendGridError && sendGridError.response) {
        console.error('[SendGrid] Error status code:', sendGridError.response.statusCode);
        console.error('[SendGrid] Error body:', sendGridError.response.body);
      }
      return false;
    }
  } catch (error) {
    console.error('[SendGrid] Email preparation error:', error);
    const sendGridError = error as any;
    if (sendGridError.response?.body?.errors) {
      console.error('[SendGrid] SendGrid detailed errors:', sendGridError.response.body.errors);
    }
    return false;
  }
}