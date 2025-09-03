import { Analysis, InsertAnalysis, Message, InsertMessage, Share, InsertShare, Session, InsertSession } from "@shared/schema";

export interface IStorage {
  // Analysis operations
  createAnalysis(analysis: InsertAnalysis): Promise<Analysis>;
  updateAnalysisPeopleCount(analysisId: number, count: number): Promise<void>;
  getAnalysisBySessionId(sessionId: string): Promise<Analysis | undefined>;
  getAnalysisById(id: number): Promise<Analysis | undefined>;
  getAllAnalysesBySessionId(sessionId: string): Promise<Analysis[]>;
  updateAnalysisDownloadStatus(analysisId: number, downloaded: boolean): Promise<void>;
  updateAnalysisTitle(analysisId: number, title: string): Promise<void>;
  deleteAnalysis(analysisId: number): Promise<void>;
  
  // Message operations
  createMessage(message: InsertMessage): Promise<Message>;
  getMessagesBySessionId(sessionId: string): Promise<Message[]>;
  getMessagesByAnalysisId(analysisId: number): Promise<Message[]>;
  
  // Share operations
  createShare(share: InsertShare): Promise<Share>;
  getShareById(id: string | number): Promise<Share | undefined>;
  updateShareStatus(shareId: number, status: "pending" | "sent" | "error"): Promise<void>;
  
  // Session operations
  createSession(session: InsertSession): Promise<Session>;
  getSessionById(sessionId: string): Promise<Session | undefined>;
  getAllSessions(): Promise<Session[]>;
  updateSessionLastActive(sessionId: string): Promise<void>;
  updateSessionName(sessionId: string, name: string): Promise<void>;
  clearSession(sessionId: string): Promise<void>;
}

export class MemStorage implements IStorage {
  private analyses: Map<number, Analysis>;
  private messages: Map<number, Message>;
  private shares: Map<number, Share>;
  private sessions: Map<string, Session>;
  private currentAnalysisId: number;
  private currentMessageId: number;
  private currentShareId: number;

  constructor() {
    this.analyses = new Map();
    this.messages = new Map();
    this.shares = new Map();
    this.sessions = new Map();
    this.currentAnalysisId = 1;
    this.currentMessageId = 1;
    this.currentShareId = 1;
  }

  async createAnalysis(insertAnalysis: InsertAnalysis): Promise<Analysis> {
    const id = this.currentAnalysisId++;
    
    // Extract peopleCount from personalityInsights if available
    let peopleCount = 1; // Default to 1 
    try {
      const insights = insertAnalysis.personalityInsights as any;
      if (insights && typeof insights === 'object' && 'peopleCount' in insights) {
        peopleCount = insights.peopleCount || 1;
      }
    } catch (err) {
      console.error("Error extracting peopleCount:", err);
    }
    
    // Make sure title is set if not provided
    const title = insertAnalysis.title || 
      (insertAnalysis.mediaType === 'image' ? 'Image Analysis' : 
        insertAnalysis.mediaType === 'video' ? 'Video Analysis' : 
        insertAnalysis.mediaType === 'document' ? 'Document Analysis' : 'Text Analysis');
    
    const analysis: Analysis = {
      id,
      sessionId: insertAnalysis.sessionId,
      title: title,
      mediaUrl: insertAnalysis.mediaUrl,
      mediaType: insertAnalysis.mediaType,
      faceAnalysis: insertAnalysis.faceAnalysis || null,
      videoAnalysis: insertAnalysis.videoAnalysis || null,
      audioTranscription: insertAnalysis.audioTranscription || null,
      documentAnalysis: insertAnalysis.documentAnalysis || null,
      textContent: insertAnalysis.textContent || null,
      personalityInsights: insertAnalysis.personalityInsights,
      peopleCount: peopleCount,
      modelUsed: insertAnalysis.modelUsed || 'openai',
      documentType: insertAnalysis.documentType || null,
      hasDownloaded: false,
      createdAt: new Date()
    };
    
    this.analyses.set(id, analysis);
    
    // Automatically create or update session
    this.updateSessionLastActive(insertAnalysis.sessionId).catch(err => {
      console.error("Error updating session active time:", err);
    });
    
    return analysis;
  }

  async getAnalysisBySessionId(sessionId: string): Promise<Analysis | undefined> {
    return Array.from(this.analyses.values()).find(
      (analysis) => analysis.sessionId === sessionId
    );
  }

  async getAnalysisById(id: number): Promise<Analysis | undefined> {
    return this.analyses.get(id);
  }
  
  async updateAnalysisPeopleCount(analysisId: number, count: number): Promise<void> {
    const analysis = this.analyses.get(analysisId);
    if (analysis) {
      this.analyses.set(analysisId, { 
        ...analysis, 
        peopleCount: count 
      });
    }
  }

  async createMessage(insertMessage: InsertMessage): Promise<Message> {
    const id = this.currentMessageId++;
    
    // Ensure analysisId is properly handled (null instead of undefined)
    const analysisId = insertMessage.analysisId !== undefined 
      ? insertMessage.analysisId 
      : null;
      
    const message: Message = {
      ...insertMessage,
      id,
      createdAt: new Date(),
      analysisId: analysisId
    };
    
    this.messages.set(id, message);
    return message;
  }

  async getMessagesBySessionId(sessionId: string): Promise<Message[]> {
    return Array.from(this.messages.values())
      .filter((message) => message.sessionId === sessionId)
      .sort((a, b) => (a.createdAt?.getTime() || 0) - (b.createdAt?.getTime() || 0));
  }

  async createShare(insertShare: InsertShare): Promise<Share> {
    const id = this.currentShareId++;
    const share: Share = {
      ...insertShare,
      id,
      status: "pending",
      createdAt: new Date(),
    };
    this.shares.set(id, share);
    return share;
  }

  async getShareById(id: string | number): Promise<Share | undefined> {
    // Convert string id to number if needed
    const numericId = typeof id === 'string' ? parseInt(id, 10) : id;
    return this.shares.get(numericId);
  }

  async updateShareStatus(shareId: number, status: "pending" | "sent" | "error"): Promise<void> {
    const share = this.shares.get(shareId);
    if (share) {
      this.shares.set(shareId, { ...share, status });
    }
  }

  // New methods for Analysis operations
  async getAllAnalysesBySessionId(sessionId: string): Promise<Analysis[]> {
    return Array.from(this.analyses.values())
      .filter(analysis => analysis.sessionId === sessionId)
      .sort((a, b) => {
        // Sort by creation date, newest first
        const dateA = a.createdAt instanceof Date ? a.createdAt : new Date();
        const dateB = b.createdAt instanceof Date ? b.createdAt : new Date();
        return dateB.getTime() - dateA.getTime();
      });
  }

  async updateAnalysisDownloadStatus(analysisId: number, downloaded: boolean): Promise<void> {
    const analysis = this.analyses.get(analysisId);
    if (analysis) {
      this.analyses.set(analysisId, {
        ...analysis,
        hasDownloaded: downloaded
      });
    }
  }

  async updateAnalysisTitle(analysisId: number, title: string): Promise<void> {
    const analysis = this.analyses.get(analysisId);
    if (analysis) {
      this.analyses.set(analysisId, {
        ...analysis,
        title
      });
    }
  }

  async deleteAnalysis(analysisId: number): Promise<void> {
    this.analyses.delete(analysisId);
    
    // Also delete any messages associated with this analysis
    const messagesToDelete = Array.from(this.messages.entries())
      .filter(([_, message]) => message.analysisId === analysisId);
    
    for (const [id] of messagesToDelete) {
      this.messages.delete(id);
    }
  }

  // Additional Message operations
  async getMessagesByAnalysisId(analysisId: number): Promise<Message[]> {
    return Array.from(this.messages.values())
      .filter(message => message.analysisId === analysisId)
      .sort((a, b) => {
        // Sort by creation date, oldest first for conversation flow
        const dateA = a.createdAt instanceof Date ? a.createdAt : new Date();
        const dateB = b.createdAt instanceof Date ? b.createdAt : new Date();
        return dateA.getTime() - dateB.getTime();
      });
  }

  // Session operations
  async createSession(session: InsertSession): Promise<Session> {
    const sessionId = session.sessionId;
    const newSession: Session = {
      id: this.sessions.size + 1,
      sessionId,
      name: session.name || "Session",
      isActive: true,
      createdAt: new Date(),
      lastActiveAt: new Date()
    };
    
    this.sessions.set(sessionId, newSession);
    return newSession;
  }

  async getSessionById(sessionId: string): Promise<Session | undefined> {
    return this.sessions.get(sessionId);
  }

  async getAllSessions(): Promise<Session[]> {
    return Array.from(this.sessions.values())
      .sort((a, b) => {
        // Sort by last active date, newest first
        const dateA = a.lastActiveAt instanceof Date ? a.lastActiveAt : new Date();
        const dateB = b.lastActiveAt instanceof Date ? b.lastActiveAt : new Date();
        return dateB.getTime() - dateA.getTime();
      });
  }

  async updateSessionLastActive(sessionId: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    
    if (session) {
      // Update existing session
      this.sessions.set(sessionId, {
        ...session,
        lastActiveAt: new Date()
      });
    } else {
      // Create new session if it doesn't exist
      const newSession: Session = {
        id: this.sessions.size + 1,
        sessionId,
        name: "Session",
        isActive: true,
        createdAt: new Date(),
        lastActiveAt: new Date()
      };
      
      this.sessions.set(sessionId, newSession);
    }
  }

  async updateSessionName(sessionId: string, name: string): Promise<void> {
    const session = this.sessions.get(sessionId);
    if (session) {
      this.sessions.set(sessionId, {
        ...session,
        name
      });
    }
  }

  async clearSession(sessionId: string): Promise<void> {
    // Get all analyses for this session
    const analyses = await this.getAllAnalysesBySessionId(sessionId);
    
    // Delete each analysis (which will also delete associated messages)
    for (const analysis of analyses) {
      await this.deleteAnalysis(analysis.id);
    }
    
    // Update session to mark it as fresh
    const session = this.sessions.get(sessionId);
    if (session) {
      this.sessions.set(sessionId, {
        ...session,
        lastActiveAt: new Date()
      });
    }
  }
}

export const storage = new MemStorage();