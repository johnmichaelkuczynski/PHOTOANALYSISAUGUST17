import * as fs from 'fs';
import * as path from 'path';
import * as os from 'os';
import { promisify } from 'util';
import { Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, BorderStyle } from 'docx';
import { Analysis } from '../../shared/schema';
import * as pdf from 'html-pdf';

const writeFileAsync = promisify(fs.writeFile);
const unlinkAsync = promisify(fs.unlink);

// Function to generate plain text format
export function generateAnalysisTxt(analysis: Analysis): string {
  const personalityInsights = analysis.personalityInsights as any || {};
  const isMultiPersonAnalysis = personalityInsights.individualProfiles && 
                              Array.isArray(personalityInsights.individualProfiles) && 
                              personalityInsights.individualProfiles.length > 1;
  
  let txtContent = '';
  
  // Header
  txtContent += '='.repeat(80) + '\n';
  txtContent += 'AI-POWERED PSYCHOLOGICAL ANALYSIS REPORT\n';
  txtContent += '='.repeat(80) + '\n\n';
  
  txtContent += `Analysis ID: ${analysis.id}\n`;
  txtContent += `Created: ${new Date(analysis.createdAt).toLocaleString()}\n`;
  txtContent += `Media Type: ${analysis.mediaType}\n`;
  txtContent += `People Detected: ${personalityInsights.peopleCount || 1}\n\n`;
  
  if (isMultiPersonAnalysis) {
    txtContent += personalityInsights.overviewSummary + '\n\n';
    
    personalityInsights.individualProfiles.forEach((profile: any, index: number) => {
      txtContent += '='.repeat(60) + '\n';
      txtContent += `INDIVIDUAL PROFILE ${index + 1}\n`;
      txtContent += '='.repeat(60) + '\n\n';
      
      txtContent += `SUMMARY:\n${profile.summary || 'No summary available'}\n\n`;
      
      const detailed = profile.detailed_analysis || {};
      
      if (detailed.cognitive_profile) {
        txtContent += 'COGNITIVE PROFILE:\n';
        txtContent += '-'.repeat(40) + '\n';
        if (detailed.cognitive_profile.intelligence_assessment) {
          txtContent += `Intelligence Assessment: ${detailed.cognitive_profile.intelligence_assessment}\n\n`;
        }
        if (detailed.cognitive_profile.cognitive_strengths) {
          txtContent += `Cognitive Strengths:\n`;
          detailed.cognitive_profile.cognitive_strengths.forEach((strength: string) => {
            txtContent += `• ${strength}\n`;
          });
          txtContent += '\n';
        }
        if (detailed.cognitive_profile.cognitive_weaknesses) {
          txtContent += `Cognitive Weaknesses:\n`;
          detailed.cognitive_profile.cognitive_weaknesses.forEach((weakness: string) => {
            txtContent += `• ${weakness}\n`;
          });
          txtContent += '\n';
        }
        if (detailed.cognitive_profile.processing_style) {
          txtContent += `Processing Style: ${detailed.cognitive_profile.processing_style}\n\n`;
        }
        if (detailed.cognitive_profile.mental_agility) {
          txtContent += `Mental Agility: ${detailed.cognitive_profile.mental_agility}\n\n`;
        }
      }
      
      if (detailed.personality_core) {
        txtContent += `PERSONALITY CORE:\n${detailed.personality_core}\n\n`;
      }
      
      if (detailed.thought_patterns) {
        txtContent += `THOUGHT PATTERNS:\n${detailed.thought_patterns}\n\n`;
      }
      
      if (detailed.emotional_intelligence) {
        txtContent += `EMOTIONAL INTELLIGENCE:\n${detailed.emotional_intelligence}\n\n`;
      }
      
      if (detailed.speech_analysis) {
        txtContent += 'SPEECH ANALYSIS:\n';
        txtContent += '-'.repeat(40) + '\n';
        
        if (detailed.speech_analysis.key_quotes && detailed.speech_analysis.key_quotes.length > 0) {
          txtContent += 'Key Quotes:\n';
          detailed.speech_analysis.key_quotes.forEach((quote: string, i: number) => {
            txtContent += `${i + 1}. "${quote}"\n`;
          });
          txtContent += '\n';
        }
        
        if (detailed.speech_analysis.vocabulary_analysis) {
          txtContent += `Vocabulary Analysis: ${detailed.speech_analysis.vocabulary_analysis}\n\n`;
        }
        
        if (detailed.speech_analysis.speech_patterns) {
          txtContent += `Speech Patterns: ${detailed.speech_analysis.speech_patterns}\n\n`;
        }
        
        if (detailed.speech_analysis.emotional_tone) {
          txtContent += `Emotional Tone: ${detailed.speech_analysis.emotional_tone}\n\n`;
        }
      }
      
      if (detailed.visual_evidence) {
        txtContent += 'VISUAL EVIDENCE:\n';
        txtContent += '-'.repeat(40) + '\n';
        
        if (detailed.visual_evidence.facial_expressions) {
          txtContent += `Facial Expressions: ${detailed.visual_evidence.facial_expressions}\n\n`;
        }
        
        if (detailed.visual_evidence.body_language) {
          txtContent += `Body Language: ${detailed.visual_evidence.body_language}\n\n`;
        }
        
        if (detailed.visual_evidence.emotional_indicators) {
          txtContent += `Emotional Indicators: ${detailed.visual_evidence.emotional_indicators}\n\n`;
        }
      }
      
      if (detailed.professional_insights) {
        txtContent += `PROFESSIONAL INSIGHTS:\n${detailed.professional_insights}\n\n`;
      }
      
      if (detailed.relationships) {
        txtContent += 'RELATIONSHIPS:\n';
        txtContent += '-'.repeat(40) + '\n';
        txtContent += `Current Status: ${detailed.relationships.current_status || 'Not specified'}\n`;
        txtContent += `Parental Status: ${detailed.relationships.parental_status || 'Not specified'}\n`;
        txtContent += `Ideal Partner: ${detailed.relationships.ideal_partner || 'Not specified'}\n\n`;
      }
      
      if (detailed.growth_areas) {
        txtContent += 'GROWTH AREAS:\n';
        txtContent += '-'.repeat(40) + '\n';
        
        if (detailed.growth_areas.strengths) {
          txtContent += 'Strengths:\n';
          detailed.growth_areas.strengths.forEach((strength: string) => {
            txtContent += `• ${strength}\n`;
          });
          txtContent += '\n';
        }
        
        if (detailed.growth_areas.challenges) {
          txtContent += 'Challenges:\n';
          detailed.growth_areas.challenges.forEach((challenge: string) => {
            txtContent += `• ${challenge}\n`;
          });
          txtContent += '\n';
        }
        
        if (detailed.growth_areas.development_path) {
          txtContent += `Development Path: ${detailed.growth_areas.development_path}\n\n`;
        }
      }
    });
  } else {
    // Single person analysis
    const profile = personalityInsights.individualProfiles?.[0] || personalityInsights;
    txtContent += `SUMMARY:\n${profile.summary || 'No summary available'}\n\n`;
    
    const detailed = profile.detailed_analysis || {};
    
    if (detailed.cognitive_profile) {
      txtContent += 'COGNITIVE PROFILE:\n';
      txtContent += '-'.repeat(40) + '\n';
      if (detailed.cognitive_profile.intelligence_assessment) {
        txtContent += `Intelligence Assessment: ${detailed.cognitive_profile.intelligence_assessment}\n\n`;
      }
      if (detailed.cognitive_profile.cognitive_strengths) {
        txtContent += `Cognitive Strengths:\n`;
        detailed.cognitive_profile.cognitive_strengths.forEach((strength: string) => {
          txtContent += `• ${strength}\n`;
        });
        txtContent += '\n';
      }
      if (detailed.cognitive_profile.cognitive_weaknesses) {
        txtContent += `Cognitive Weaknesses:\n`;
        detailed.cognitive_profile.cognitive_weaknesses.forEach((weakness: string) => {
          txtContent += `• ${weakness}\n`;
        });
        txtContent += '\n';
      }
      if (detailed.cognitive_profile.processing_style) {
        txtContent += `Processing Style: ${detailed.cognitive_profile.processing_style}\n\n`;
      }
      if (detailed.cognitive_profile.mental_agility) {
        txtContent += `Mental Agility: ${detailed.cognitive_profile.mental_agility}\n\n`;
      }
    }
    
    // Continue with other sections like personality_core, thought_patterns, etc.
    if (detailed.personality_core) {
      txtContent += `PERSONALITY CORE:\n${detailed.personality_core}\n\n`;
    }
    
    if (detailed.thought_patterns) {
      txtContent += `THOUGHT PATTERNS:\n${detailed.thought_patterns}\n\n`;
    }
    
    if (detailed.emotional_intelligence) {
      txtContent += `EMOTIONAL INTELLIGENCE:\n${detailed.emotional_intelligence}\n\n`;
    }
    
    if (detailed.speech_analysis) {
      txtContent += 'SPEECH ANALYSIS:\n';
      txtContent += '-'.repeat(40) + '\n';
      
      if (detailed.speech_analysis.key_quotes && detailed.speech_analysis.key_quotes.length > 0) {
        txtContent += 'Key Quotes:\n';
        detailed.speech_analysis.key_quotes.forEach((quote: string, i: number) => {
          txtContent += `${i + 1}. "${quote}"\n`;
        });
        txtContent += '\n';
      }
      
      if (detailed.speech_analysis.vocabulary_analysis) {
        txtContent += `Vocabulary Analysis: ${detailed.speech_analysis.vocabulary_analysis}\n\n`;
      }
      
      if (detailed.speech_analysis.speech_patterns) {
        txtContent += `Speech Patterns: ${detailed.speech_analysis.speech_patterns}\n\n`;
      }
      
      if (detailed.speech_analysis.emotional_tone) {
        txtContent += `Emotional Tone: ${detailed.speech_analysis.emotional_tone}\n\n`;
      }
    }
    
    if (detailed.visual_evidence) {
      txtContent += 'VISUAL EVIDENCE:\n';
      txtContent += '-'.repeat(40) + '\n';
      
      if (detailed.visual_evidence.facial_expressions) {
        txtContent += `Facial Expressions: ${detailed.visual_evidence.facial_expressions}\n\n`;
      }
      
      if (detailed.visual_evidence.body_language) {
        txtContent += `Body Language: ${detailed.visual_evidence.body_language}\n\n`;
      }
      
      if (detailed.visual_evidence.emotional_indicators) {
        txtContent += `Emotional Indicators: ${detailed.visual_evidence.emotional_indicators}\n\n`;
      }
    }
    
    if (detailed.professional_insights) {
      txtContent += `PROFESSIONAL INSIGHTS:\n${detailed.professional_insights}\n\n`;
    }
    
    if (detailed.relationships) {
      txtContent += 'RELATIONSHIPS:\n';
      txtContent += '-'.repeat(40) + '\n';
      txtContent += `Current Status: ${detailed.relationships.current_status || 'Not specified'}\n`;
      txtContent += `Parental Status: ${detailed.relationships.parental_status || 'Not specified'}\n`;
      txtContent += `Ideal Partner: ${detailed.relationships.ideal_partner || 'Not specified'}\n\n`;
    }
    
    if (detailed.growth_areas) {
      txtContent += 'GROWTH AREAS:\n';
      txtContent += '-'.repeat(40) + '\n';
      
      if (detailed.growth_areas.strengths) {
        txtContent += 'Strengths:\n';
        detailed.growth_areas.strengths.forEach((strength: string) => {
          txtContent += `• ${strength}\n`;
        });
        txtContent += '\n';
      }
      
      if (detailed.growth_areas.challenges) {
        txtContent += 'Challenges:\n';
        detailed.growth_areas.challenges.forEach((challenge: string) => {
          txtContent += `• ${challenge}\n`;
        });
        txtContent += '\n';
      }
      
      if (detailed.growth_areas.development_path) {
        txtContent += `Development Path: ${detailed.growth_areas.development_path}\n\n`;
      }
    }
  }
  
  txtContent += '='.repeat(80) + '\n';
  txtContent += 'END OF REPORT\n';
  txtContent += '='.repeat(80) + '\n';
  
  return txtContent;
}

// Function to generate HTML for PDF
export function generateAnalysisHtml(analysis: Analysis): string {
  // Extract the personality insights
  const personalityInsights = analysis.personalityInsights as any || {};
  
  // Determine if we have a multi-person analysis
  const isMultiPersonAnalysis = personalityInsights.individualProfiles && 
                              Array.isArray(personalityInsights.individualProfiles) && 
                              personalityInsights.individualProfiles.length > 1;
  
  // Get total people count
  const peopleCount = personalityInsights.peopleCount || 1;
  
  // For backward compatibility with single-person analysis
  let summary = 'No summary available';
  let detailedAnalysis = {} as any;

  // Format content differently based on single vs. multiple people
  let htmlContent = `
    <html>
    <head>
      <style>
        body {
          font-family: Arial, sans-serif;
          line-height: 1.6;
          color: #333;
          max-width: 800px;
          margin: 0 auto;
          padding: 20px;
        }
        h1 {
          color: #2563eb;
          border-bottom: 1px solid #ddd;
          padding-bottom: 10px;
        }
        h2 {
          color: #4338ca;
          margin-top: 25px;
        }
        h3 {
          color: #1e40af;
        }
        .summary {
          background-color: #f9fafb;
          border-left: 4px solid #3b82f6;
          padding: 15px;
          margin: 20px 0;
        }
        .profile {
          background-color: #f3f4f6;
          border-radius: 8px;
          padding: 15px;
          margin: 20px 0;
          box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .section {
          margin: 15px 0;
        }
        .footer {
          margin-top: 40px;
          font-size: 0.8em;
          color: #6b7280;
          text-align: center;
        }
      </style>
    </head>
    <body>
      <h1>${analysis.title || 'Personality Analysis Report'}</h1>
      <p><strong>Date:</strong> ${new Date().toLocaleDateString()}</p>
      <p><strong>Analysis Type:</strong> ${analysis.mediaType.charAt(0).toUpperCase() + analysis.mediaType.slice(1)} Analysis</p>
  `;
  
  if (isMultiPersonAnalysis) {
    // Handle multi-person analysis
    const profiles = personalityInsights.individualProfiles || [];
    const overviewSummary = personalityInsights.overviewSummary || `Analysis of ${peopleCount} people detected in the media.`;
    
    htmlContent += `
      <div class="summary">
        <h2>Overview (${peopleCount} People Detected)</h2>
        <p>${overviewSummary}</p>
      </div>
    `;
    
    if (personalityInsights.groupDynamics) {
      htmlContent += `
        <div class="section">
          <h2>Group Dynamics</h2>
          <p>${personalityInsights.groupDynamics}</p>
        </div>
      `;
    }
    
    // Add individual profiles section
    htmlContent += `<h2>Individual Profiles</h2>`;
    
    // Add each person's profile
    profiles.forEach((profile: any, index: number) => {
      const personLabel = profile.personLabel || `Person ${index + 1}`;
      const personSummary = profile.summary || 'No summary available';
      const personDetails = profile.detailed_analysis || {};
      
      htmlContent += `
        <div class="profile">
          <h3>${personLabel}</h3>
          
          <div class="section">
            <h4>Summary</h4>
            <p>${personSummary}</p>
          </div>
          
          <div class="section">
            <h4>Core Personality</h4>
            <p>${personDetails.personality_core || 'Not available'}</p>
          </div>
          
          <div class="section">
            <h4>Thought Patterns</h4>
            <p>${personDetails.thought_patterns || 'Not available'}</p>
          </div>
          
          <div class="section">
            <h4>Professional Insights</h4>
            <p>${personDetails.professional_insights || 'Not available'}</p>
          </div>
      `;
      
      if (personDetails.growth_areas) {
        htmlContent += `
          <div class="section">
            <h4>Key Strengths</h4>
            <ul>
              ${Array.isArray(personDetails.growth_areas.strengths) 
                ? personDetails.growth_areas.strengths.map((s: string) => `<li>${s}</li>`).join('') 
                : '<li>Not available</li>'}
            </ul>
            
            <h4>Challenges</h4>
            <ul>
              ${Array.isArray(personDetails.growth_areas.challenges) 
                ? personDetails.growth_areas.challenges.map((c: string) => `<li>${c}</li>`).join('') 
                : '<li>Not available</li>'}
            </ul>
          </div>
        `;
      }
      
      htmlContent += `</div>`;
    });
    
  } else {
    // Get data for single-person format
    if (personalityInsights.individualProfiles?.length === 1) {
      const profile = personalityInsights.individualProfiles[0];
      summary = profile.summary || 'No summary available';
      detailedAnalysis = profile.detailed_analysis || {};
    } else {
      // Legacy format
      summary = personalityInsights.summary || 'No summary available';
      detailedAnalysis = personalityInsights.detailed_analysis || {};
    }
    
    // Generate single-person content
    htmlContent += `
      <div class="summary">
        <h2>Summary</h2>
        <p>${summary}</p>
      </div>

      <div class="section">
        <h2>Core Personality Traits</h2>
        <p>${detailedAnalysis.personality_core || 'Not available'}</p>
      </div>

      <div class="section">
        <h2>Thought Patterns</h2>
        <p>${detailedAnalysis.thought_patterns || 'Not available'}</p>
      </div>

      <div class="section">
        <h2>Cognitive Style</h2>
        <p>${detailedAnalysis.cognitive_style || 'Not available'}</p>
      </div>

      <div class="section">
        <h2>Professional Insights</h2>
        <p>${detailedAnalysis.professional_insights || 'Not available'}</p>
      </div>
    `;
    
    if (detailedAnalysis.growth_areas) {
      htmlContent += `
        <div class="section">
          <h2>Growth Areas</h2>
          
          <h3>Strengths</h3>
          <ul>
            ${(detailedAnalysis.growth_areas.strengths || []).map((s: string) => `<li>${s}</li>`).join('')}
          </ul>
          
          <h3>Challenges</h3>
          <ul>
            ${(detailedAnalysis.growth_areas.challenges || []).map((c: string) => `<li>${c}</li>`).join('')}
          </ul>
          
          <h3>Development Path</h3>
          <p>${detailedAnalysis.growth_areas.development_path || 'Not available'}</p>
        </div>
      `;
    }
  }
  
  htmlContent += `
      <div class="footer">
        <p>Generated by Personality Insights App</p>
        <p>${new Date().toISOString()}</p>
      </div>
    </body>
    </html>
  `;
  
  return htmlContent;
}

// Generate DOCX document for an analysis
export async function generateDocx(analysis: Analysis): Promise<Buffer> {
  // Extract the personality insights
  const personalityInsights = analysis.personalityInsights as any || {};
  
  // Determine if we have a multi-person analysis
  const isMultiPersonAnalysis = personalityInsights.individualProfiles && 
                              Array.isArray(personalityInsights.individualProfiles) && 
                              personalityInsights.individualProfiles.length > 1;
  
  // Get total people count
  const peopleCount = personalityInsights.peopleCount || 1;
  
  // Document content
  const doc = new Document({
    title: analysis.title || 'Personality Analysis Report',
    description: 'AI-generated personality analysis',
    sections: [{
      properties: {},
      children: []
    }],
  });

  const children = [];

  // Title
  children.push(
    new Paragraph({
      text: analysis.title || 'Personality Analysis Report',
      heading: HeadingLevel.HEADING_1,
      alignment: AlignmentType.CENTER,
    })
  );

  // Date and Type
  children.push(
    new Paragraph({
      children: [
        new TextRun({ text: 'Date: ', bold: true }),
        new TextRun(new Date().toLocaleDateString()),
      ],
    })
  );

  children.push(
    new Paragraph({
      children: [
        new TextRun({ text: 'Analysis Type: ', bold: true }),
        new TextRun(analysis.mediaType.charAt(0).toUpperCase() + analysis.mediaType.slice(1)),
      ],
    })
  );

  children.push(new Paragraph({})); // Spacing

  if (isMultiPersonAnalysis) {
    // Handle multi-person analysis
    const profiles = personalityInsights.individualProfiles || [];
    const overviewSummary = personalityInsights.overviewSummary || `Analysis of ${peopleCount} people detected in the media.`;
    
    // Overview
    children.push(
      new Paragraph({
        text: `Overview (${peopleCount} People Detected)`,
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: overviewSummary }));
    
    // Group Dynamics if available
    if (personalityInsights.groupDynamics) {
      children.push(
        new Paragraph({
          text: 'Group Dynamics',
          heading: HeadingLevel.HEADING_2,
        })
      );
      
      children.push(new Paragraph({ text: personalityInsights.groupDynamics }));
    }
    
    // Individual Profiles
    children.push(
      new Paragraph({
        text: 'Individual Profiles',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    // Add each person's profile
    profiles.forEach((profile: any, index: number) => {
      const personLabel = profile.personLabel || `Person ${index + 1}`;
      const personSummary = profile.summary || 'No summary available';
      const personDetails = profile.detailed_analysis || {};
      
      children.push(
        new Paragraph({
          text: personLabel,
          heading: HeadingLevel.HEADING_3,
        })
      );
      
      // Summary
      children.push(
        new Paragraph({
          children: [new TextRun({ text: 'Summary', bold: true })],
        })
      );
      
      children.push(new Paragraph({ text: personSummary }));
      
      // Core Personality
      children.push(
        new Paragraph({
          children: [new TextRun({ text: 'Core Personality', bold: true })],
        })
      );
      
      children.push(new Paragraph({ text: personDetails.personality_core || 'Not available' }));
      
      // Thought Patterns
      children.push(
        new Paragraph({
          children: [new TextRun({ text: 'Thought Patterns', bold: true })],
        })
      );
      
      children.push(new Paragraph({ text: personDetails.thought_patterns || 'Not available' }));
      
      // Professional Insights
      children.push(
        new Paragraph({
          children: [new TextRun({ text: 'Professional Insights', bold: true })],
        })
      );
      
      children.push(new Paragraph({ text: personDetails.professional_insights || 'Not available' }));
      
      // Strengths and Challenges if available
      if (personDetails.growth_areas) {
        children.push(
          new Paragraph({
            children: [new TextRun({ text: 'Key Strengths', bold: true })],
          })
        );
        
        if (Array.isArray(personDetails.growth_areas.strengths)) {
          personDetails.growth_areas.strengths.forEach((strength: string) => {
            children.push(new Paragraph({ text: `• ${strength}` }));
          });
        } else {
          children.push(new Paragraph({ text: '• Not available' }));
        }
        
        children.push(
          new Paragraph({
            children: [new TextRun({ text: 'Challenges', bold: true })],
          })
        );
        
        if (Array.isArray(personDetails.growth_areas.challenges)) {
          personDetails.growth_areas.challenges.forEach((challenge: string) => {
            children.push(new Paragraph({ text: `• ${challenge}` }));
          });
        } else {
          children.push(new Paragraph({ text: '• Not available' }));
        }
      }
      
      // Add spacing after each profile
      children.push(new Paragraph({}));
    });
    
  } else {
    // Get data for single-person format
    let summary = 'No summary available';
    let detailedAnalysis = {} as any;
    
    if (personalityInsights.individualProfiles?.length === 1) {
      const profile = personalityInsights.individualProfiles[0];
      summary = profile.summary || 'No summary available';
      detailedAnalysis = profile.detailed_analysis || {};
    } else {
      // Legacy format
      summary = personalityInsights.summary || 'No summary available';
      detailedAnalysis = personalityInsights.detailed_analysis || {};
    }
    
    // Summary
    children.push(
      new Paragraph({
        text: 'Summary',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: summary }));
    
    // Core Personality
    children.push(
      new Paragraph({
        text: 'Core Personality Traits',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: detailedAnalysis.personality_core || 'Not available' }));
    
    // Thought Patterns
    children.push(
      new Paragraph({
        text: 'Thought Patterns',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: detailedAnalysis.thought_patterns || 'Not available' }));
    
    // Cognitive Style
    children.push(
      new Paragraph({
        text: 'Cognitive Style',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: detailedAnalysis.cognitive_style || 'Not available' }));
    
    // Professional Insights
    children.push(
      new Paragraph({
        text: 'Professional Insights',
        heading: HeadingLevel.HEADING_2,
      })
    );
    
    children.push(new Paragraph({ text: detailedAnalysis.professional_insights || 'Not available' }));
    
    // Growth Areas if available
    if (detailedAnalysis.growth_areas) {
      children.push(
        new Paragraph({
          text: 'Growth Areas',
          heading: HeadingLevel.HEADING_2,
        })
      );
      
      // Strengths
      children.push(
        new Paragraph({
          text: 'Strengths',
          heading: HeadingLevel.HEADING_3,
        })
      );
      
      if (Array.isArray(detailedAnalysis.growth_areas.strengths)) {
        detailedAnalysis.growth_areas.strengths.forEach((strength: string) => {
          children.push(new Paragraph({ text: `• ${strength}` }));
        });
      } else {
        children.push(new Paragraph({ text: '• Not available' }));
      }
      
      // Challenges
      children.push(
        new Paragraph({
          text: 'Challenges',
          heading: HeadingLevel.HEADING_3,
        })
      );
      
      if (Array.isArray(detailedAnalysis.growth_areas.challenges)) {
        detailedAnalysis.growth_areas.challenges.forEach((challenge: string) => {
          children.push(new Paragraph({ text: `• ${challenge}` }));
        });
      } else {
        children.push(new Paragraph({ text: '• Not available' }));
      }
      
      // Development Path
      children.push(
        new Paragraph({
          text: 'Development Path',
          heading: HeadingLevel.HEADING_3,
        })
      );
      
      children.push(new Paragraph({ text: detailedAnalysis.growth_areas.development_path || 'Not available' }));
    }
  }
  
  // Footer
  children.push(new Paragraph({})); // Spacing
  children.push(
    new Paragraph({
      text: 'Generated by Personality Insights App',
      alignment: AlignmentType.CENTER,
    })
  );
  
  children.push(
    new Paragraph({
      text: new Date().toISOString(),
      alignment: AlignmentType.CENTER,
    })
  );
  
  // Create a new document with proper structure
  const newDoc = new Document({
    sections: [{
      properties: {},
      children: children
    }]
  });
  
  return await Packer.toBuffer(newDoc);
}

// Create PDF from HTML content
export async function generatePdf(htmlContent: string): Promise<Buffer> {
  return new Promise<Buffer>((resolve, reject) => {
    const options = {
      format: 'Letter',
      border: {
        top: '0.5in',
        right: '0.5in',
        bottom: '0.5in',
        left: '0.5in'
      }
    };
    
    pdf.create(htmlContent, options).toBuffer((err, buffer) => {
      if (err) {
        reject(err);
        return;
      }
      if (!buffer) {
        reject(new Error("Failed to generate PDF: Empty buffer"));
        return;
      }
      resolve(buffer);
    });
  });
}