declare module 'html-pdf' {
  export interface PdfOptions {
    format?: string;
    orientation?: 'portrait' | 'landscape';
    border?: {
      top?: string;
      right?: string;
      bottom?: string;
      left?: string;
    } | string;
    header?: {
      height?: string;
      contents?: string;
    };
    footer?: {
      height?: string;
      contents?: string;
    };
    zoomFactor?: number;
    type?: string;
    quality?: number;
    renderDelay?: number;
    script?: string;
    timeout?: number;
    phantomPath?: string;
    phantomArgs?: string[];
    localUrlAccess?: boolean;
  }

  export function create(html: string, options?: PdfOptions): {
    toFile(path: string, callback: (err: Error | null, res?: any) => void): void;
    toBuffer(callback: (err: Error | null, buffer?: Buffer) => void): void;
    toStream(callback: (err: Error | null, stream?: NodeJS.ReadableStream) => void): void;
  };
}