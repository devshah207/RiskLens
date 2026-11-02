import React from 'react';
import { createRoot } from 'react-dom/client';
import type { FrontendRenderer } from '@streamlit/component-v2-lib';
import { Explorer } from './Explorer';
import type { StoriesPayload } from './types';
import '@xyflow/react/dist/style.css';
import './styles.css';

/** Streamlit mounts the component in its page (no external iframe or CDN). */
const render: FrontendRenderer = ({ parentElement, data }) => {
  const host = parentElement.querySelector('#risklens-reactflow-root');
  if (!host) return;
  const root = createRoot(host);
  root.render(<React.StrictMode><Explorer payload={data as StoriesPayload} /></React.StrictMode>);
  return () => root.unmount();
};

export default render;
