export type MobileTab = "queue" | "map" | "card" | "system";

interface Props {
  tab: MobileTab;
  attention: number;
  selected: string | null;
  onTab: (tab: MobileTab) => void;
}

/** Phone navigation: one panel at a time (hidden by CSS on wider screens). */
export function MobileTabs({ tab, attention, selected, onTab }: Props) {
  const items: { id: MobileTab; label: string; note?: string }[] = [
    { id: "queue", label: "Очередь", note: attention > 0 ? String(attention) : undefined },
    { id: "map", label: "Карта" },
    { id: "card", label: selected ? `ТС ${selected}` : "Карточка" },
    { id: "system", label: "Система" },
  ];
  return (
    <nav className="mobile-tabs" aria-label="Разделы пульта">
      {items.map((item) => (
        <button key={item.id} type="button" className={tab === item.id ? "active" : ""}
          aria-current={tab === item.id ? "page" : undefined} onClick={() => onTab(item.id)}>
          <span>{item.label}</span>
          {item.note ? <b className="mobile-tab-count" aria-label={`требуют внимания: ${item.note}`}>{item.note}</b> : null}
        </button>
      ))}
    </nav>
  );
}
