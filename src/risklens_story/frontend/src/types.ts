export type NodeKind = 'source' | 'guard' | 'process' | 'model' | 'evidence' | 'governance';

export interface StoryNode {
  [key: string]: unknown;
  id: string;
  title: string;
  subtitle: string;
  position: { x: number; y: number };
  plain: string;
  technical: string;
  evidence: string;
  sourcePath: string;
  kind: NodeKind;
}

export interface StoryEdge {
  id: string;
  source: string;
  target: string;
  label: string;
}

export interface Story {
  id: 'market' | 'research';
  title: string;
  subtitle: string;
  nodes: StoryNode[];
  edges: StoryEdge[];
  journey: string[];
  footer: string;
}

export interface StoriesPayload {
  market: Story;
  research: Story;
  context: {
    savedHoldoutVerified: boolean;
    publicRepository: string;
    notes: string;
  };
}
