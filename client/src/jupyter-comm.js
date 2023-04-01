// import JSON5 from 'json5';
import autoBind from 'auto-bind';
import defer from 'lodash.defer';
import debounce from 'lodash.debounce';

export default class JupyterComm {
  constructor(interfaceId, onopen) {
    autoBind(this);
    this.interfaceId = interfaceId;
    this.callbackMap = {};
    this.data = {};
    this.pendingData = {};
    this.jcomm = new InnerJupyterComm('guidance_interface_target_'+this.interfaceId, this.updateData, "open");

    this.debouncedSendPendingData500 = debounce(this.sendPendingData, 500);
    this.debouncedSendPendingData1000 = debounce(this.sendPendingData, 1000);
    if (onopen) {
      defer(onopen);
    }
  }

  send(keys, data) {
    this.addPendingData(keys, data);
    this.sendPendingData();
  }

  sendEvent(commEvent) {
    for (const k of Object.keys(commEvent)) {
      this.addPendingData(k, commEvent[k]);
    }
    this.sendPendingData();
  }

  debouncedSendEvent500(commEvent) {
    for (const k of Object.keys(commEvent)) {
      this.addPendingData(k, commEvent[k]);
    }
    this.debouncedSendPendingData500();
  }

  debouncedSend500(keys, data) {
    this.addPendingData(keys, data);
    this.debouncedSendPendingData500();
  }

  debouncedSend1000(keys, data) {
    this.addPendingData(keys, data);
    this.debouncedSendPendingData1000();
  }

  addPendingData(keys, data) {

    // console.log("addPendingData", keys, data);
    if (!Array.isArray(keys)) keys = [keys];
    for (const i in keys) this.pendingData[keys[i]] = data;
  }

  updateData(data) {
    data = JSON.parse(data["data"]) // data from Jupyter is wrapped so we get to do our own JSON encoding
    // console.log("updateData", data)

    // save the data locally
    for (const k in data) {
      this.data[k] = data[k];
    }

    // call all the registered callbacks
    for (const k in data) {
      if (k in this.callbackMap) {
        this.callbackMap[k](this.data[k]);
      }
    }
  }

  subscribe(key, callback) {
    this.callbackMap[key] = callback;
    defer(_ => this.callbackMap[key](this.data[key]));
  }

  sendPendingData() {
    // console.log("sending", this.pendingData);
    this.jcomm.send_data(this.pendingData);
    this.pendingData = {};
  }
}

class InnerJupyterComm {
  constructor(target_name, callback, mode="open") {
    this._fire_callback = this._fire_callback.bind(this);
    this._register = this._register.bind(this)

    this.jcomm = undefined;
    this.callback = callback;

    // check if the Jupyter variable is defined
    if (window.Jupyter !== undefined) {
      // https://jupyter-notebook.readthedocs.io/en/stable/comms.html
      if (mode === "register") {
        Jupyter.notebook.kernel.comm_manager.register_target(target_name, this._register);
      } else {
        // debugger;
        this.jcomm = Jupyter.notebook.kernel.comm_manager.new_comm(target_name);
        this.jcomm.on_msg(this._fire_callback);
        // console.log("target_name", target_name, this.jcomm)
      }
    } else if (window._mgr !== undefined) {
      
      // this.jcomm = window._mgr.widgetManager.proxyKernel.createComm(target_name);
      // this.jcomm.onMsg(this._fire_callback);
      if (mode === "register") {
        window._mgr.widgetManager.proxyKernel.registerCommTarget(target_name, this._register)
        // console.error("register not supported in vscode");
        //Jupyter.notebook.kernel.comm_manager.register_target(target_name, this._register);
      } else {
        // debugger;
        // console.log("create comm in vscode", target_name)
        // WritableStreamDefaultController.asdfk.sd.f.sdf
        this.jcomm = window._mgr.widgetManager.proxyKernel.createComm(target_name);
        this.jcomm.open({}, "");

        this.jcomm.onMsg = this._fire_callback;
        // this.jcomm = Jupyter.notebook.kernel.comm_manager.new_comm(target_name);
        // this.jcomm.on_msg(this._fire_callback);
        // console.log("target_name", target_name, this.jcomm)
      }
    }
  }

  send_data(data) {
    if (this.jcomm !== undefined) {
      this.jcomm.send(data);
    } else {
      console.error("Jupyter comm module not yet loaded! So we can't send the message.")
    }
  }

  _register(jcomm, msg) {
    this.jcomm = jcomm;
    this.jcomm.on_msg(this._fire_callback);
  }

  _fire_callback(msg) {
    // console.log("_fire_callback", msg)
    this.callback(msg.content.data)
  }
}

// const comm = JupyterComm();

// // Jupyter.notebook.kernel.comm_manager.register_target('gadfly_comm_target',
// //   function(jcomm, msg) {
// //     // comm is the frontend comm instance
// //     // msg is the comm_open message, which can carry data

// //     comm.jcomm = jcomm

// //     // Register handlers for later messages:
// //     inner_comm.on_msg(function(msg) { console.log("MSGG", msg); });
// //     inner_comm.on_close(function(msg) { console.log("MSGdG", msg); });
// //     comm.send({'foo': 0});
// //   }
// // );

// export default comm;