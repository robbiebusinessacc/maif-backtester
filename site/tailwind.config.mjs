/** @type {import('tailwindcss').Config} */
export default {
  content: ["./src/**/*.{astro,html,md,mdx,ts,tsx,js,jsx}"],
  theme: {
    extend: {
      colors: {
        primary: {
          DEFAULT: "#881C1C",
          dark: "#5F1111",
        },
        ink: "#0F0F0F",
        paper: "#FAF7F2",
        muted: "#8B8680",
        hair: "#E8E4DE",
        positive: "#2E7D32",
        negative: "#C62828",
        aux: "#7B6D5C",
      },
      fontFamily: {
        display: ['"Cormorant Garamond"', "ui-serif", "Georgia", "serif"],
        body: ['"Inter"', "system-ui", "sans-serif"],
        mono: ['"JetBrains Mono"', "ui-monospace", "SFMono-Regular", "monospace"],
      },
      fontVariantNumeric: {
        tabular: "tabular-nums",
      },
      maxWidth: {
        prose: "44rem",
        wide: "72rem",
      },
      letterSpacing: {
        eyebrow: "0.18em",
      },
    },
  },
  plugins: [],
};
