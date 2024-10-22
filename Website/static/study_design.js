var nodes_des = null;
var edges_des = null;
var network_des = null;
var network_des_popup = null;

function edgeExists(edges, from, to) {
    return edges.some(edge => edge.from === from && edge.to === to);
}

function draw_design(js_design_data, div_id, colors) {
    nodes_des = [];
    edges_des = [];

    let x_positions = {
        "Intervention": 0,
        "Observation": 0,
        "Count": 0,
        "Outcome": 0,
    };

    let y_positions = {
        "Intervention": 100,
        "Observation": 200,
        "Count": 200,
        "Outcome": 300,
    };

    let type_counts = {
        "Intervention": 0,
        "Observation": 0,
        "Count": 0,
        "Outcome": 0,
    };

    for (let i = 0; i < js_design_data['study_design'].length; i++) {
        type_counts[js_design_data['study_design'][i]['type']] += 1;
    }

    let x_steps = {
        "Intervention": 1000 / (type_counts["Intervention"] + 1) + 10,
        "Observation": 1000 / (type_counts["Observation"] + type_counts['Count'] + 1) + 10,
        "Count": 1000 / (type_counts["Observation"] + type_counts['Count'] + 1) + 10,
        "Outcome": 1000 / (type_counts["Outcome"] + 1) + 10,
    };

    // draw network
    for (let i = 0; i < js_design_data['study_design'].length; i++) {
        let node = js_design_data['study_design'][i];
        let node_id = node['label'];
        let node_type = node['type'];
        let node_label = node['term'];

        x_positions[node_type] += x_steps[node_type];
        if (node_type === "Observation") {
            x_positions['Count'] += x_steps['Count'];
        }
        if (node_type === "Count") {
            x_positions['Observation'] += x_steps['Observation'];
        }
        nodes_des.push({
            id: node_id,
            label: node_label,
            x: x_positions[node_type],
            y: y_positions[node_type],
            color: colors[node_type],
        });

        for (let j = 0; j < node['connects_from'].length; j++) {
            let from_id = node['connects_from'][j];
            if (from_id === "Root") {
                continue;
            }
            if (!edgeExists(edges_des, from_id, node_id)) {
                edges_des.push({from: from_id, to: node_id, color: 'gray'});
            }
        }
        for (let j = 0; j < node['connects_to'].length; j++) {
            let to_id = node['connects_to'][j];
            if (!edgeExists(edges_des, node_id, to_id)) {
                edges_des.push({from: node_id, to: to_id, color: 'gray'});
            }
        }
    }

    // create container and network
    var data_des = {
        nodes: nodes_des,
        edges: edges_des,
    };

    var options = {
        edges: {
            font: {
                size: 12,
            },
            widthConstraint: {
                maximum: 90,
            },
            arrows: {to: true},
        },
        nodes: {
            shape: "box",
            font: {
                size: 14,
            },
            margin: 10,
            widthConstraint: {
                maximum: 200,
            },
        },
        physics: {
            enabled: false,
        },
        layout: {
            improvedLayout: true,
            //hierarchical:{blockShifting: true},
        }
    };

    if (div_id === "study_design") {
        var container_des = document.getElementById("study_design");
        if (js_design_data['study_design'].length === 0) {
            container_des.innerHTML = "<p style='display: flex; justify-content: center; align-items: center; height: 100%;'>Unable to infer study design.</p>";
        } else {
            network_des = new vis.Network(container_des, data_des, options);
        }
    } else if (div_id === "study_design_popup") {
        var container_des_popup = document.getElementById("study_design_popup");
        if (js_design_data['study_design'].length === 0) {
            container_des_popup.innerHTML = "<p style='display: flex; justify-content: center; align-items: center; height: 100%;'>Unable to infer study design.</p>";
        } else {
            options.nodes.font.size = 14;
            network_des_popup = new vis.Network(container_des_popup, data_des, options);
        }
    }
}


// Get the modal
var modal_des = document.getElementById("myModal_design");

// Get the button that opens the modal
var btn_des = document.getElementById("myBtn_design");

// Get the <span> element that closes the modal
var span_des = document.getElementsByClassName("close")[1];

btn_des.onclick = function () {
    modal_des.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
span_des.onclick = function () {
    modal_des.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
// window.onclick = function(event) {
//   if (event.target == modal_des) {
//     modal_des.style.display = "none";
//   }
// }

//window.onresize = function() {network_des_popup.fit();}
//network_des_popup.fit();
// network_des_popup.moveTo({
//       position: {x:-500, y:-300}
//     });

// window.addEventListener("load", () => {
//   draw_design();
// });

