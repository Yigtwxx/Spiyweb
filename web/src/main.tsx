import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

// Fonts are bundled, never fetched from a CDN: this tool runs on a laptop
// that may be offline, and a measurement rig should not depend on the network
// to render its own numbers.
import "@fontsource/ibm-plex-sans/400.css";
import "@fontsource/ibm-plex-sans/500.css";
import "@fontsource/ibm-plex-sans/600.css";
import "@fontsource/ibm-plex-sans-condensed/500.css";
import "@fontsource/ibm-plex-sans-condensed/600.css";
import "@fontsource/ibm-plex-mono/400.css";
import "@fontsource/ibm-plex-mono/500.css";
import "@fontsource/eb-garamond/400-italic.css";

import "./styles/theme.css";
import "./styles/base.css";
import "./styles/plate.css";
import App from "./App";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>,
);
