import { useEffect, useMemo, useRef, useState } from 'react';
import type { KeyboardEvent } from 'react';
import {
  Background,
  BackgroundVariant,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  useReactFlow,
  ReactFlowProvider,
  useNodesInitialized,
} from '@xyflow/react';
import type { Edge, Node, NodeProps, NodeTypes } from '@xyflow/react';
import type { StoriesPayload, Story, StoryNode } from './types';

const ICON: Record<string, string> = {
  source: '◈', guard: '◆', process: '⌁', model: '✳', evidence: '◎', governance: '✦',
};

const SOURCE_ROOT = 'https://github.com/devshah207/RiskLens/blob/main/';
const AUTO_INTERVAL_MS = 6500;
const NODE_WIDTH = 298;
const NODE_HEIGHT = 154;
const GUIDED_ZOOM = 1.13;
const POSITION_SCALE_X = 1.40;
const POSITION_SCALE_Y = 1.48;

type DisplayNode = Node<StoryNode & Record<string, unknown> & { active: boolean; visited: boolean }, 'story'>;

function GraphNode({ data: rawData }: NodeProps) {
  const data = rawData as DisplayNode['data'];
  const cls = `risk-node kind-${data.kind}${data.active ? ' is-active' : ''}${data.visited ? ' is-visited' : ''}`;
  return (
    <div className={cls} aria-label={`${data.title}: ${data.subtitle}`}>
      <Handle type="target" position={Position.Left} isConnectable={false} />
      <span className="node-icon" aria-hidden="true">{ICON[data.kind]}</span>
      <span className="node-kind">{data.kind}</span>
      <strong className="node-title">{data.title}</strong>
      <span className="node-subtitle">{data.subtitle}</span>
      <Handle type="source" position={Position.Right} isConnectable={false} />
    </div>
  );
}

const NODE_TYPES: NodeTypes = { story: GraphNode };

function safeSourcePath(path: string): boolean {
  return (path.startsWith('src/risklens/') || path.startsWith('app/'))
    && !path.split('/').includes('..') && !path.includes('?') && !path.includes('#');
}

function useReducedMotion(): boolean {
  const [reduced, setReduced] = useState(() =>
    typeof window !== 'undefined' && window.matchMedia('(prefers-reduced-motion: reduce)').matches,
  );
  useEffect(() => {
    const query = window.matchMedia('(prefers-reduced-motion: reduce)');
    const onChange = () => setReduced(query.matches);
    query.addEventListener('change', onChange);
    return () => query.removeEventListener('change', onChange);
  }, []);
  return reduced;
}

/** Keep the current stage readable instead of squeezing the whole seven-node map
 * into the initial viewport. Wait until React Flow has measured nodes before
 * placing the camera: an immediate setCenter can be overwritten by fitView.
 */
function FitSelected({ selectedId, signal }: { selectedId: string; signal: number }) {
  const flow = useReactFlow();
  const initialized = useNodesInitialized();
  const reduced = useReducedMotion();
  useEffect(() => {
    if (!initialized) return;
    let secondFrame = 0;
    const firstFrame = window.requestAnimationFrame(() => {
      secondFrame = window.requestAnimationFrame(() => {
        const node = flow.getNode(selectedId);
        if (node) {
          void flow.setCenter(node.position.x + NODE_WIDTH / 2,
            node.position.y + NODE_HEIGHT / 2,
            { zoom: GUIDED_ZOOM, duration: reduced ? 0 : 480 });
        }
      });
    });
    return () => {
      window.cancelAnimationFrame(firstFrame);
      window.cancelAnimationFrame(secondFrame);
    };
  }, [selectedId, initialized, flow, reduced, signal]);
  return null;
}

function ExplorerContent({ payload }: { payload: StoriesPayload }) {
  const [track, setTrack] = useState<'market' | 'research'>('market');
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [technical, setTechnical] = useState(false);
  const [explore, setExplore] = useState(false);
  const [detailsOpen, setDetailsOpen] = useState(true);
  const [focusMode, setFocusMode] = useState(false);
  const [fullscreenActive, setFullscreenActive] = useState(false);
  const [cameraSignal, setCameraSignal] = useState(0);
  const shellRef = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();
  const flow = useReactFlow();

  // A component-v2 UI lives inside a ShadowRoot. The document may retarget
  // fullscreenElement to the shadow host, so check the local root as well.
  const isOurFullscreen = () => {
    const root = shellRef.current?.getRootNode();
    const rootElement = root && 'fullscreenElement' in root
      ? (root as Document | ShadowRoot).fullscreenElement
      : null;
    return rootElement === shellRef.current || document.fullscreenElement === shellRef.current;
  };

  // Fullscreen is requested ONLY by a direct user click. Browser Escape exits it.
  // Browsers that deny the API still get a large in-page presentation layout.
  useEffect(() => {
    const syncFullscreen = () => {
      const isActive = isOurFullscreen();
      setFullscreenActive(isActive);
      if (!document.fullscreenElement) setFocusMode(false);
    };
    document.addEventListener('fullscreenchange', syncFullscreen);
    return () => document.removeEventListener('fullscreenchange', syncFullscreen);
  }, []);

  const togglePresentation = async () => {
    if (focusMode) {
      if (isOurFullscreen() && document.exitFullscreen) {
        try { await document.exitFullscreen(); } catch { /* Exit via browser Escape */ }
      }
      setFocusMode(false);
      return;
    }
    setDetailsOpen(false);
    setFocusMode(true);
    if (shellRef.current?.requestFullscreen) {
      try {
        await shellRef.current.requestFullscreen();
      } catch {
        // The in-page immersive layout remains usable without fullscreen access.
      }
    }
    setCameraSignal((v) => v + 1);
  };
  const story: Story = payload[track];
  const selectedId = story.journey[Math.min(step, story.journey.length - 1)];
  const selected = story.nodes.find((n) => n.id === selectedId) ?? story.nodes[0];

  useEffect(() => {
    if (!playing || reduced) return;
    const timer = window.setInterval(() => {
      setStep((old) => Math.min(story.journey.length - 1, old + 1));
    }, AUTO_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [playing, story.journey.length, track, reduced]);

  useEffect(() => {
    if (playing && step >= story.journey.length - 1) setPlaying(false);
  }, [step, playing, story.journey.length]);

  // Stop automatic motion if the operating system enables reduced motion.
  useEffect(() => {
    if (reduced) setPlaying(false);
  }, [reduced]);

  const nodes: DisplayNode[] = useMemo(() => story.nodes.map((node) => ({
    id: node.id,
    type: 'story' as const,
    position: {
      x: node.position.x * POSITION_SCALE_X,
      y: node.position.y * POSITION_SCALE_Y,
    },
    data: { ...node, active: node.id === selectedId,
      visited: story.journey.indexOf(node.id) < step },
    draggable: false,
    selectable: true,
    zIndex: node.id === selectedId ? 12 : 1,
    ariaLabel: `Open ${node.title}`,
  })), [story, selectedId, step]);

  const edges: Edge[] = useMemo(() => story.edges.map((edge) => {
    const highlights = edge.source === selectedId || edge.target === selectedId;
    return {
      ...edge,
      type: 'smoothstep',
      animated: highlights && !reduced,
      selectable: false,
      label: edge.label || undefined,
      style: { stroke: highlights ? '#55e7c2' : '#365065',
        strokeWidth: highlights ? 4.2 : 2.35, opacity: highlights ? 1 : 0.69 },
      labelStyle: { fill: '#d5e7f5', fontSize: 14, fontWeight: 650 },
      labelBgStyle: { fill: '#142238', fillOpacity: 0.95 },
    };
  }), [story, selectedId, reduced]);

  const chooseStory = (next: 'market' | 'research') => {
    setTrack(next);
    setStep(0);
    setPlaying(false);
    setExplore(false);
    if (!focusMode) setDetailsOpen(true);
    setCameraSignal((v) => v + 1);
  };
  const previous = () => { setPlaying(false); setStep((v) => Math.max(0, v - 1)); };
  const next = () => { setPlaying(false); setStep((v) => Math.min(story.journey.length - 1, v + 1)); };
  const replay = () => { setStep(0); setPlaying(false); setExplore(false); setCameraSignal((v) => v + 1); };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    const target = event.target as HTMLElement;
    if (['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
    if (event.key === 'Escape' && focusMode && !fullscreenActive) {
      event.preventDefault(); setFocusMode(false); return;
    }
    if (event.key === 'ArrowRight') { event.preventDefault(); next(); }
    if (event.key === 'ArrowLeft') { event.preventDefault(); previous(); }
  };
  const selectNode = (id: string) => {
    const i = story.journey.indexOf(id);
    if (i !== -1) { setStep(i); setPlaying(false); setExplore(true); }
  };

  return (
    <div ref={shellRef}
      className={`explorer-shell${focusMode ? ' is-presentation' : ''}${fullscreenActive ? ' is-fullscreen' : ''}`}
      onKeyDown={onKeyDown} tabIndex={0}
      aria-label="Interactive RiskLens architecture explorer. Use left and right arrows for stages.">
      <div className="explorer-topline">
        <span className="explorer-brand"><span className="brand-icon">◉</span> RiskLens <span className="brand-sub">/ INTERACTIVE ARCHITECTURE</span></span>
        <span className="live-indicator"><i /> READ-ONLY WALKTHROUGH</span>
      </div>
      <div className="story-heading">
        <div>
          <div className="eyebrow">SYSTEM STORY / CHOOSE YOUR JOURNEY</div>
          <h2>Go beyond the dashboard.<br /><span>Understand the engineering.</span></h2>
          <p>Follow the real processing decisions, inspect every stage, and see where the evidence comes from.</p>
        </div>
        <div className="story-switch" role="group" aria-label="Select architecture">
          <button className={track === 'market' ? 'active' : ''} onClick={() => chooseStory('market')}
            aria-pressed={track === 'market'}>◉ Market risk</button>
          <button className={track === 'research' ? 'active' : ''} onClick={() => chooseStory('research')}
            aria-pressed={track === 'research'}>◇ Financial RAG</button>
        </div>
      </div>

      <div className="story-progress" aria-label="Narrative stages">
        {story.journey.map((id, i) => {
          const node = story.nodes.find((n) => n.id === id);
          return (
            <button key={id} className={`story-progress-step ${i === step ? 'active' : ''} ${i < step ? 'done' : ''}`}
              onClick={() => { setStep(i); setPlaying(false); }}
              aria-current={i === step ? 'step' : undefined} aria-label={`Stage ${i + 1}: ${node?.title}`}>
              <span className="progress-bar" />
              <span className="progress-number">{String(i + 1).padStart(2, '0')}</span>
              <span className="progress-name">{node?.title}</span>
            </button>
          );
        })}
      </div>

      <div className={`flow-layout${!detailsOpen ? ' details-collapsed' : ''}`}>
        <section className="flow-surface" aria-label={`${story.title} interactive node graph`}>
          <div className="flow-label">
            <div className="flow-label-title"><span className="flow-dot" /> ARCHITECTURE CANVAS <small>Drag to pan · Scroll to zoom · Click any node</small></div>
            <div className="flow-actions" role="group" aria-label="Diagram display controls">
              <button type="button" onClick={() => void flow.fitView({ padding: 0.1, duration: reduced ? 0 : 450 })}
                title="Overview of the full architecture">▣ View whole map</button>
              <button type="button" onClick={() => setCameraSignal((v) => v + 1)}
                title="Magnify the currently selected stage">◎ Focus stage</button>
              <button type="button" onClick={() => { setDetailsOpen((v) => !v); setCameraSignal((v) => v + 1); }}
                aria-expanded={detailsOpen} aria-controls="story-stage-details">
                {detailsOpen ? '◧ Hide details' : '◨ Show details'}
              </button>
              <button type="button" className="presentation-button" onClick={() => void togglePresentation()}
                aria-pressed={focusMode}>
                {focusMode ? '↙ Exit presentation' : '⛶ Presentation mode'}
              </button>
            </div>
          </div>
          <ReactFlow
            nodes={nodes}
            edges={edges}
            nodeTypes={NODE_TYPES}
            onNodeClick={(_, node) => selectNode(node.id)}
            nodesDraggable={false}
            nodesConnectable={false}
            edgesFocusable={false}
            edgesReconnectable={false}
            zoomOnDoubleClick={false}
            fitView
            fitViewOptions={{ padding: 0.12 }}
            onInit={() => setCameraSignal((v) => v + 1)}
            minZoom={0.18}
            maxZoom={1.8}
            colorMode="dark"
            proOptions={{ hideAttribution: false }}
          >
            <Background color="#2d4356" gap={27} size={1} variant={BackgroundVariant.Dots} />
            <MiniMap nodeStrokeWidth={3} nodeColor="#23556b" maskColor="rgba(5,12,23,.70)" pannable zoomable />
            <Controls showInteractive={false} />
            <FitSelected selectedId={selectedId} signal={cameraSignal} />
          </ReactFlow>
          <div className="flow-footer"><span>{story.footer}</span><span>Original Python pipeline preserved</span></div>
        </section>

        {detailsOpen && <aside id="story-stage-details" className="detail-panel" aria-live="polite">
          <div className="detail-header">
            <div><span className="eyebrow">STAGE {String(step + 1).padStart(2, '0')} / {story.journey.length}</span>
              <h3>{selected.title}</h3>
              <p>{selected.subtitle}</p>
            </div>
            <span className={`detail-icon detail-${selected.kind}`}>{ICON[selected.kind]}</span>
          </div>
          <div className="detail-mode" role="group" aria-label="Explanation level">
            <button className={!technical ? 'active' : ''} aria-pressed={!technical} onClick={() => setTechnical(false)}>✦ Simple</button>
            <button className={technical ? 'active' : ''} aria-pressed={technical} onClick={() => setTechnical(true)}>⌘ Technical</button>
          </div>
          <div className="detail-explanation"><div className="eyebrow">{technical ? 'UNDER THE HOOD' : 'IN PLAIN ENGLISH'}</div>
            <p>{technical ? selected.technical : selected.plain}</p>
          </div>
          <div className="detail-proof"><div className="eyebrow">WHAT THE EVIDENCE REALLY SAYS</div>
            <p>{selected.evidence}</p>
          </div>
          <div className="detail-bottom">
            <div className="eyebrow">EXPLORE ACTUAL IMPLEMENTATION</div>
            {safeSourcePath(selected.sourcePath) ? (
              <a href={SOURCE_ROOT + selected.sourcePath} target="_blank" rel="noopener noreferrer" className="source-button">
                <span>↗</span><span>View original Python code <small>{selected.sourcePath}</small></span>
              </a>
            ) : <span className="source-button">Code reference unavailable</span>}
            <p className="trace-note">This visualization reads published architecture metadata. It does not execute a model or call Groq.</p>
          </div>
        </aside>}
      </div>

      <div className="explorer-bottom">
        <div className="playback-controls">
          <button className="round-button" onClick={previous} disabled={step === 0} title="Previous stage" aria-label="Previous stage">←</button>
          <button className="primary-control" onClick={() => {
            if (playing) setPlaying(false);
            else if (reduced) { next(); }
            else { if (step >= story.journey.length - 1) setStep(0); setPlaying(true); setExplore(false); }
          }} aria-label={reduced ? 'Advance stage' : playing ? 'Pause guided tour' : 'Play guided tour'}>
            {reduced ? 'Next step →' : playing ? 'Ⅱ Pause tour' : '▶ Play guided tour'}
          </button>
          <button className="round-button" onClick={next} disabled={step === story.journey.length - 1} title="Next stage" aria-label="Next stage">→</button>
          <button className="ghost-control" onClick={replay} title="Replay from first stage">↺ Replay</button>
        </div>
        <div className="playback-context">
          <strong>{explore ? 'Explore mode' : playing ? 'Guided tour playing' : 'Self-paced tour'}</strong>
          <span>{reduced ? 'Reduced motion enabled' : '6.5s per stage · click any node'} · {focusMode ? 'Presentation view' : 'Explorer view'}</span>
        </div>
      </div>
      <div className="explorer-disclosure">
        <span>◇</span>
        <p><strong>Designed to explain, not embellish.</strong> The market diagram uses frozen historical research and the XBRL/PDF diagram shows alternative retrieval paths. It is not a live trading simulation, new evaluation or a certified financial answer.</p>
      </div>
    </div>
  );
}

export function Explorer({ payload }: { payload: StoriesPayload }) {
  if (!payload || !payload.market || !payload.research) {
    return <p>Architecture metadata is unavailable. No data is being simulated.</p>;
  }
  return <ReactFlowProvider><ExplorerContent payload={payload} /></ReactFlowProvider>;
}
