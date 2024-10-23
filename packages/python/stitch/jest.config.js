module.exports = {
  automock: false,
  moduleNameMapper: {
    '\\.(css|less|sass|scss)$': 'identity-obj-proxy',
  },
  preset: 'ts-jest/presets/js-with-babel',
  moduleFileExtensions: ['ts', 'tsx', 'js', 'jsx', 'json', 'node'],
  testPathIgnorePatterns: ['/lib/', '/node_modules/'],
  testRegex: '/__tests__/.*.spec.ts[x]?$',
  transformIgnorePatterns: ['/node_modules/(?!(@jupyter(lab|-widgets)/.*)/)'],
  globals: {
    'ts-jest': {
      tsconfig: '<rootDir>/tsconfig.json',
    },
  },
};
