/**
 * Global toast queue. Any screen can push a toast (attendance mark, enroll
 * success, new critical incident); ToastHost renders + auto-dismisses them.
 */
import { create } from 'zustand';

export type ToastTone = 'success' | 'info' | 'alert';

export type Toast = {
  id: string;
  text: string;
  tone: ToastTone;
  /** Optional Ionicons glyph name. */
  icon?: string;
};

type ToastState = {
  toasts: Toast[];
  show: (text: string, tone?: ToastTone, icon?: string) => void;
  dismiss: (id: string) => void;
};

let counter = 0;

export const useToast = create<ToastState>((set) => ({
  toasts: [],
  show: (text, tone = 'info', icon) => {
    counter += 1;
    const id = `t${counter}`;
    set((s) => ({ toasts: [...s.toasts, { id, text, tone, icon }] }));
  },
  dismiss: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
}));

/** Non-hook accessor for use outside React (e.g. the ws handler). */
export const toast = {
  show: (text: string, tone?: ToastTone, icon?: string) =>
    useToast.getState().show(text, tone, icon),
};
