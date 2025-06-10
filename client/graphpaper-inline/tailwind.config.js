/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./src/**/*.{html,ts,js,svelte}"],
  theme: {
    extend: {
      fontFamily: {
        'token': ['JetBrains Mono'],
      },
      keyframes: {
        'cpulse': {
          '50%': { opacity: 0.0 }
        }
      },
      animation: {
        'cpulse': 'cpulse 3.5s cubic-bezier(0.4, 0, 0.6, 1) infinite'
      }
    }
  },
  plugins: [
    // require('tailwind-scrollbar'),
  ],
}

