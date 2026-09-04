// Metro config for the web export.
//
// Default minification renames/strips class names. That's invisible almost
// everywhere — except `expo-modules-core`'s `registerWebModule()` (used by
// expo-blur, expo-camera, expo-image-picker, expo-secure-store, ...) keys each
// module off `moduleImplementation.name`, and throws "Module implementation
// must be a class" the moment that name is empty. On the previous default
// config this happened during the very first bundle evaluation, before React
// ever mounted — a blank white page with no on-screen error.
const { getDefaultConfig } = require('expo/metro-config');

const config = getDefaultConfig(__dirname);

config.transformer.minifierConfig = {
  ...config.transformer.minifierConfig,
  keep_classnames: true,
  keep_fnames: true,
  mangle: {
    ...(config.transformer.minifierConfig?.mangle ?? {}),
    keep_classnames: true,
    keep_fnames: true,
  },
};

module.exports = config;
