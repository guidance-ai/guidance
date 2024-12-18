// Copyright (c) Guidance Contributors
// Distributed under the terms of the Modified BSD License.

import {
  DOMWidgetModel,
  DOMWidgetView,
  ISerializers,
} from '@jupyter-widgets/base';

import { MODULE_NAME, MODULE_VERSION } from './version';

// Import the CSS
import '../css/widget.css';

export class StitchModel extends DOMWidgetModel {
  defaults() {
    return {
      ...super.defaults(),
      _model_name: StitchModel.model_name,
      _model_module: StitchModel.model_module,
      _model_module_version: StitchModel.model_module_version,
      _view_name: StitchModel.view_name,
      _view_module: StitchModel.view_module,
      _view_module_version: StitchModel.view_module_version,
      kernelmsg: '',
      clientmsg: '',
      srcdoc: '<p>srcdoc should be defined by the user</p>',
      initial_height: '1px',
      initial_width: '1px',
      initial_border: '0',
    };
  }

  static serializers: ISerializers = {
    ...DOMWidgetModel.serializers,
    // Add any extra serializers here
  };

  static model_name = 'StitchModel';
  static model_module = MODULE_NAME;
  static model_module_version = MODULE_VERSION;
  static view_name = 'StitchView'; // Set to null if no view
  static view_module = MODULE_NAME; // Set to null if no view
  static view_module_version = MODULE_VERSION;
}

export class StitchView extends DOMWidgetView {
  private _iframe: HTMLIFrameElement;

  render() {
    // Create sandboxed frame
    const iframe = document.createElement('iframe');
    iframe.sandbox.add('allow-scripts');
    iframe.srcdoc = this.model.get('srcdoc');
    iframe.style.height = this.model.get('initial_height');
    iframe.style.width = this.model.get('initial_width');
    iframe.style.border = this.model.get('initial_border');
    iframe.style.display = 'block';
    this._iframe = iframe;

    // Send first kernelmsg on load.
    const refreshTimeMs = 100;
    const sendKernelMsgOnReady = () => {
      if (this.model.isNew()) {
        window.setTimeout(sendKernelMsgOnReady, refreshTimeMs);
      } else {
        this.kernelmsg_changed();
      }
    };
    window.setTimeout(sendKernelMsgOnReady, refreshTimeMs);

    // Add callback for forwarding messages from client to kernel
    const model = this.model;
    const recvFromClient = (event: any) => {
      const win = iframe.contentWindow;
      if (win === event.source && event.data.type === 'clientmsg') {
        model.set('clientmsg', event.data.content);
        model.save_changes();
      } else if (win === event.source && event.data.type === 'resize') {
        iframe.style.height = event.data.content.height;
        iframe.style.width = event.data.content.width;
      }
    };
    window.addEventListener('message', recvFromClient);

    // Add iframe to root element of widget
    this.el.appendChild(this._iframe);

    // Connect callbacks for forwarding from kernel to client
    this.model.on('change:kernelmsg', this.kernelmsg_changed, this);

    // Connect callback for when kernel changes the srcdoc
    this.model.on('change:srcdoc', this.srcdoc_changed, this);
  }

  kernelmsg_changed() {
    // Forward message from kernel to client
    const kernelmsg = this.model.get('kernelmsg');
    const winmsg = {
      type: 'kernelmsg',
      content: kernelmsg,
    };
    this._iframe.contentWindow?.postMessage(winmsg, '*');
  }

  srcdoc_changed() {
    // Apply srcdoc provided by client
    const srcdoc = this.model.get('srcdoc');
    this._iframe.srcdoc = srcdoc;

    // Push last kernelmsg immediately to iframe
    this.kernelmsg_changed();
  }
}
