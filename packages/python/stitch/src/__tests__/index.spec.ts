/** @jest-environment jsdom */
// Copyright (c) Jupyter Development Team.
// Distributed under the terms of the Modified BSD License.

// Add any needed widget imports here (or from controls)
// import {} from '@jupyter-widgets/base';

// NOTE(nopdive): Workaround for jsdom drag event failure.
Object.defineProperty(window, 'DragEvent', {
  value: class DragEvent {},
});

import { createTestModel } from './utils';
import { StitchModel } from '..';

describe('Example', () => {
  describe('StitchModel', () => {
    it('should be createable', () => {
      const model = createTestModel(StitchModel);
      expect(model).toBeInstanceOf(StitchModel);
      expect(model.get('srcdoc')).toEqual(
        '<p>srcdoc should be defined by the user</p>',
      );
    });

    it('should be createable with a value', () => {
      const state = { srcdoc: 'it is alright' };
      const model = createTestModel(StitchModel, state);
      expect(model).toBeInstanceOf(StitchModel);
      expect(model.get('srcdoc')).toEqual('it is alright');
    });
  });
});
