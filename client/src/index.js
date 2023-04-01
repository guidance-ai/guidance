// import JSON5 from 'json5';

import PromptDisplay from './prompt-display';
// import JupyterComm from './jupyter-comm';

function _guidanceDisplay(object_id, executing) {

    // if (!window._guidanceComms) {
    //     window._guidanceComms = {};
    // }
    // window._guidanceComms[object_id] = new JupyterComm(object_id);

    // generate a display id
    // var display_id = object_id;// + "____" + Math.random().toString(36).substring(7);

    return new PromptDisplay(object_id, executing);

    // return display.element;
    /* This creates an output div element and build a comm object to communicate. */
    // var output = document.createElement('div');
    // output.id = object_id;
    // output.style.width = '100%';
    // output.style.height = '100%';
    // output.style.overflow = 'auto';

    // // put text in the dic
    // var text = document.createElement('p');
    // text.innerHTML = 'Loading...';
    // output.appendChild(text);

    // var comm = new JupyterComm(object_id);
    // comm.open();
    // return output;
}

window._guidanceDisplay = _guidanceDisplay