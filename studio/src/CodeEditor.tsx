import { json } from "@codemirror/lang-json";
import { basicSetup, EditorView } from "codemirror";
import { useEffect, useRef } from "react";

interface Props {
  label: string;
  value: string;
  onChange?: (value: string) => void;
  readOnly?: boolean;
}

export function CodeEditor({ label, value, onChange, readOnly = false }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const view = useRef<EditorView | null>(null);
  const onChangeRef = useRef(onChange);
  const valueRef = useRef(value);
  onChangeRef.current = onChange;
  valueRef.current = value;

  useEffect(() => {
    if (!host.current) return;
    const editor = new EditorView({
      doc: valueRef.current,
      parent: host.current,
      extensions: [
        basicSetup,
        json(),
        EditorView.editable.of(!readOnly),
        EditorView.theme({
          "&": { background: "#091310", color: "#d7e6de", height: "100%" },
          ".cm-content": { caretColor: "#f0bd66", fontFamily: "var(--mono)", fontSize: "12px" },
          ".cm-gutters": { background: "#091310", color: "#66766e", border: "none" },
          ".cm-activeLine, .cm-activeLineGutter": { background: "#11221c" },
          ".cm-selectionBackground, ::selection": { background: "#285845 !important" },
          ".cm-scroller": { overflow: "auto" },
        }),
        EditorView.updateListener.of((update) => {
          if (update.docChanged) onChangeRef.current?.(update.state.doc.toString());
        }),
      ],
    });
    view.current = editor;
    return () => editor.destroy();
  }, [readOnly]);

  useEffect(() => {
    const editor = view.current;
    if (!editor || editor.state.doc.toString() === value) return;
    editor.dispatch({ changes: { from: 0, to: editor.state.doc.length, insert: value } });
  }, [value]);

  return (
    <div className="code-editor" aria-label={label} role="region">
      <span className="sr-only">{label}</span>
      <div ref={host} className="code-editor-host" />
    </div>
  );
}
