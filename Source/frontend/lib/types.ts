
export interface ChunkData {
    x: number;
    y: number;
    width: number;
    height: number;
    predicted_label: string;
    probability: number;
  }
  
  export interface ClassificationData {
    predicted_label?: string;
    probability?: number;
    result_image?: string;
    chunks?: ChunkData[];
    summary?: {
      bleached_chunks: number;
      healthy_chunks: number;
      total_chunks: number;
    };
    completed_chunks?: number;
    total_chunks?: number;
    progress?: string;
    status?: string;
  }
  
  export interface ClassificationResultProps {
    file: File;
    result: ClassificationData;
  }