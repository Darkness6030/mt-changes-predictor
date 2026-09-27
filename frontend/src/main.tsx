import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import { AuthGate } from "./components/AuthGate";
import "./styles.css";

createRoot(document.getElementById("root") as HTMLElement).render(
  <StrictMode>
    <AuthGate>{(user, logout) => <App user={user} onLogout={logout} />}</AuthGate>
  </StrictMode>,
);
