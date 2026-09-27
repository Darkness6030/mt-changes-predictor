import { type FormEvent, type ReactNode, useCallback, useEffect, useState } from "react";
import { api, authToken } from "../api";
import type { AuthState } from "../types";

interface Props {
  children: (user: AuthState, logout: () => void) => ReactNode;
}

/**
 * Optional sign-in. Without accounts on the Backend (the default) the console opens at once;
 * with ``BACKEND_AUTH_USERS`` it asks for a token and shows who acknowledges alerts.
 */
export function AuthGate({ children }: Props) {
  const [user, setUser] = useState<AuthState | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const [token, setToken] = useState("");
  const [checking, setChecking] = useState(false);

  const check = useCallback(async () => {
    setChecking(true);
    try {
      setUser(await api.auth());
      setFailure(null);
    } catch {
      // No connection yet (or an older Backend): open the console as before; its banners
      // report the connection, and a later 401 brings this gate back.
      setUser({ enabled: false, authenticated: false, can_act: true });
    } finally {
      setChecking(false);
    }
  }, []);

  useEffect(() => { void check(); }, [check]);
  useEffect(() => {
    const again = () => { void check(); };
    window.addEventListener("auth-required", again);
    return () => window.removeEventListener("auth-required", again);
  }, [check]);

  const logout = useCallback(() => {
    authToken.clear();
    void check();
  }, [check]);

  const submit = async (event: FormEvent) => {
    event.preventDefault();
    authToken.set(token.trim());
    setChecking(true);
    try {
      const answer = await api.auth();
      if (!answer.authenticated) {
        authToken.clear();
        setFailure("Токен не подошёл. Проверьте его у администратора стенда.");
      } else {
        setToken("");
        setFailure(null);
      }
      setUser(answer);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : String(error));
    } finally {
      setChecking(false);
    }
  };

  if (user === null) return <div className="auth-screen"><p className="empty">Подключение…</p></div>;
  if (user.enabled && !user.authenticated) {
    return (
      <div className="auth-screen">
        <form className="auth-form" onSubmit={submit}>
          <h1>Пульт диспетчера</h1>
          <p className="hint">Вход по токену. Диспетчер отмечает алерты и управляет источником,
            наблюдатель только просматривает.</p>
          <label>
            Токен доступа
            <input type="password" autoComplete="current-password" value={token} required
              onChange={(event) => setToken(event.target.value)} autoFocus />
          </label>
          <button type="submit" disabled={checking || !token.trim()}>{checking ? "Проверка…" : "Войти"}</button>
          {failure ? <p className="auth-error" role="alert">{failure}</p> : null}
        </form>
      </div>
    );
  }
  return <>{children(user, logout)}</>;
}
