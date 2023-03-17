import JupyterComm from './jupyter-comm';
import autoBind from 'auto-bind'; 

export default class PromptDisplay {
  constructor(id) {
    autoBind(this);
    // make a new div
    this.element = document.getElementById("guidance_container_"+id);
    this.element.id = id;
    // this.the_window = window
    // window.inner_window = window;
    // // window.parent.inner_window = window;
    // console.log("window", this.the_window)

    // console.log("id", id)

    // window._mgr.postOffice.addHandler({
    //     // eslint-disable-next-line @typescript-eslint/no-explicit-any
    //     handleMessage: (type, payload) => {
    //         console.log("handle", type, payload);
    //     }
    // });
    // fill in the div
    // var str = "";
    // for (var k in window._mgr.postOffice) {
    //   str += "<br><br>" + k;
    //   console.log(k);
    //   // for (var l in window[k]) {
    //   //   str += "<br>\n" + l + ": ";// + window[k][l];
    //   // }
    // }
    // this.element.innerHTML = str;

    // create a comm object
    this.comm = new JupyterComm(id);

    this.comm.subscribe("add_data", this.newData);
  }

  newData(data) {
    // console.log("data", data);

    // check if add_data is a key
    if (data) {
      this.element.innerHTML = data;
    }
  }
}