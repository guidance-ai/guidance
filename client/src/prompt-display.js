import JupyterComm from './jupyter-comm';
import autoBind from 'auto-bind'; 

export default class PromptDisplay {
  constructor(id, element, cell) {
    autoBind(this);
    // make a new divs
    this.element = element;//document.getElementById("guidance_container_"+id);
    this.element.id = id;

    this.cell = cell;

    this.cell_element = window.$(this.element).parents('.cell');
    // which cell is it?
    // var cell_idx = Jupyter.notebook.get_cell_elements().index(cell_element);
    // // get the cell object
    // var cell = Jupyter.notebook.get_cell(cell_idx);
    console.log("cell", this.element);
    window.gelement = this.element;
    console.log("cell", this.cell_element);
    console.log("cell", this.cell_element.data());
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

    // create a comm object
    this.comm = new JupyterComm(id);

    this.comm.subscribe("add_data", this.newData);
    this.comm.subscribe("set_data", this.setData);
  }

  newData(data) {
    // console.log("data", data);

    // check if add_data is a key
    if (data) {
      this.element.innerHTML = data;
      // this.cell.outputs[0].data["text/plain"] = "ASDFASDFASDFASDFASD";
      // this.cell.outputs[0].data["text/html"] = data;
      this.cell.outputs[0].data = {
        "text/plain": "ASDFASDFASDFASDFASD",
        "text/html": data
      };
    }
  }

  setData(data) {
    console.log("data", data);

    // check if add_data is a key
    if (data) {
      this.element.innerHTML = data;
      this.cell.outputs[0].data = {
        // "text/plain": "ASDFASDFASDFASDFASD",
        "text/html": data
      };
    }
  }
}