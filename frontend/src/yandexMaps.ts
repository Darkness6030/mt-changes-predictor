let loading: Promise<typeof ymaps> | null = null;

/** One loader for all mounts (including React StrictMode), with bounded failure and retry. */
export function loadYandexMaps(): Promise<typeof ymaps> {
  if (loading) return loading;
  const key = (import.meta.env.VITE_YANDEX_MAPS_API_KEY as string | undefined)?.trim();
  if (!key) return Promise.reject(new Error(
    "Для карты нужен ключ Яндекс Карт. Добавьте VITE_YANDEX_MAPS_API_KEY в .env и пересоберите UI.",
  ));
  loading = new Promise((resolve, reject) => {
    const script = document.createElement("script");
    const fail = () => {
      clearTimeout(timer);
      script.remove();
      loading = null;
      reject(new Error("Яндекс Карты недоступны. Проверьте подключение и API-ключ. Очередь и управление временем работают."));
    };
    const timer = window.setTimeout(fail, 15000);
    script.src = `https://api-maps.yandex.ru/2.1/?apikey=${encodeURIComponent(key)}&lang=ru_RU`;
    script.async = true;
    script.onerror = fail;
    script.onload = () => {
      if (typeof ymaps === "undefined") { fail(); return; }
      ymaps.ready(() => { clearTimeout(timer); resolve(ymaps); });
    };
    document.head.appendChild(script);
  });
  return loading;
}

/** API hintContent is HTML; telemetry and plan addresses must remain plain text. */
export function hintHtml(text: string): string {
  return text.replace(/[&<>"']/g, (char) => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
  })[char]!).replace(/\n/g, "<br>");
}

export function svgImage(svg: string): string {
  return `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svg)}`;
}
