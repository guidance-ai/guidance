import JupyterComm from './jupyter-comm';
import autoBind from 'auto-bind'; 

export default class PromptDisplay {
  constructor(id, executing) {
    autoBind(this);
    // make a new divs
    // this.element = element;//document.getElementById("guidance_container_"+id);
    this.id = id;

    // create a comm object
    this.comm = new JupyterComm(id);
    this.comm.subscribe("append", this.appendData);
    this.comm.subscribe("replace", this.replaceData);
    this.comm.subscribe("event", this.eventOccurred);

    this.element = document.getElementById("guidance-content-"+id);
    this.stop_button = document.getElementById("guidance-stop-button-"+id);
    this.stop_button.onclick = () => this.comm.send("event", "stop");
    // this.stop_button = document.createElement('div');
    // this.stop_button.style.cssText = 'cursor: pointer; margin: 0px; display: none; float: right; padding: 3px; border-radius: 4px 4px 4px 4px; border: 0px solid rgba(127, 127, 127, 1); padding-left: 10px; padding-right: 10px; font-size: 13px; background-color: rgba(127, 127, 127, 0.25);';
    // 
    // this.stop_button.innerText = 'Stop program';
    // if (!executing) {
    //   this.stop_button.style.display = "none";
    // }

    // add the stop button as the first child of the element
    // this.element.parentNode.insertBefore(this.stop_button, this.element);

    // this.cell = cell;

    // this.cell_element = window.$(this.element).parents('.cell');
    // which cell is it?
    // var cell_idx = Jupyter.notebook.get_cell_elements().index(cell_element);
    // // get the cell object
    // var cell = Jupyter.notebook.get_cell(cell_idx);
    // console.log("cell", this.element);
    // window.gelement = this.element;
    // console.log("cell", this.cell_element);
    // console.log("cell", this.cell_element.data());
    // debugger;
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

    
  }

  appendData(data) {
    // console.log("appendData", data);

    if (data) {
      this.stop_button.style.display = "inline-block";
      this.element.innerHTML += data;
      // this.cell.outputs[0].data["text/plain"] = "ASDFASDFASDFASDFASD";
      // this.cell.outputs[0].data["text/html"] = data;
      
      // this sets the data that will be saved in the notebook file (only works in Jupyter notebook, not JupyterLab or VSCode)
      // this.cell.outputs[0].data = {
      //   "text/html": data
      // };
    }
  }

  replaceData(data) {
    // console.log("replaceData", data);

    if (data) {
      this.stop_button.style.display = "inline-block";
      this.element.innerHTML = data;

      // this sets the data that will be saved in the notebook file (only works in Jupyter notebook, not JupyterLab or VSCode)
      // this.cell.outputs[0].data = {
      //   "text/html": data
      // };
    }
  }

  eventOccurred(name) {
    // console.log("complete", name);
    if (name === "complete") {
      this.stop_button.style.display = "none";
    }
  }
}