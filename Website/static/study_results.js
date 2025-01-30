var nodes_res = null;
var edges_res = null;
var network_results = null;
var network_results_popup = null;

function edgeExists(edges, from, to) {
    return edges.some(edge => edge.from === from && edge.to === to);
}

function draw_study_result(js_results_data, div_id, colors) {
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

    for (let i = 0; i < js_results_data['study_results'].length; i++) {
        type_counts[js_results_data['study_results'][i]['type']] += 1;
    }

    let x_steps = {
        "Intervention": 1000 / (type_counts["Intervention"] + 1) + 10,
        "Observation": 1000 / (type_counts["Observation"] + type_counts['Count'] + 1) + 10,
        "Count": 1000 / (type_counts["Observation"] + type_counts['Count'] + 1) + 10,
        "Outcome": 1000 / (type_counts["Outcome"] + 1) + 10,
    };

    // draw network
    let idMapping = {};
    for (let i = 0; i < js_results_data['study_results'].length; i++) {
        let node = js_results_data['study_results'][i];
        let node_id
        if (div_id === "collective_visualization") {
            node_id = `doc${Math.floor(i/6)}_node${i}_${node['label']}`;
            idMapping[node['label']] = node_id;
        } else {
            node_id = node['label']; 
        }
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
            if (div_id === "collective_visualization") {
                // Update the from_id to match the new node ID format
                from_id = idMapping[from_id];
            }
            if (!edgeExists(edges_des, from_id, node_id)) {
                edges_des.push({from: from_id, to: node_id, color: 'gray'});
            }
        }
        for (let j = 0; j < node['connects_to'].length; j++) {
            let to_id = node['connects_to'][j];
            if (div_id === "collective_visualization") {
                // Update the to_id to match the new node ID format
                to_id = idMapping[to_id];
            }
            if (!edgeExists(edges_des, node_id, to_id)) {
                edges_des.push({from: node_id, to: to_id, color: 'gray'});
            }
        }
    }


    // create container and network
    var data_res = {
        nodes: nodes_des,
        edges: edges_des,
    };
    var options = {
        edges: {
            font: {
                size: 16,
            },
            widthConstraint: {
                maximum: 90,
            },
            arrows: {to: true},
        },
        nodes: {
            shape: "box",
            margin: 10,
            font: {
                size: 16,
            },
            widthConstraint: {
                maximum: 150,
            },
        },
        physics: {
            enabled: false,
        },
    };

    if (div_id === "study_results") {
        var container_res = document.getElementById("study_results");
        if (js_results_data['study_results'].length === 0) {
            container_res.innerHTML = "<p style='display: flex; justify-content: center; align-items: center; height: 100%;'>No study results found</p>";
        } else {
            network_results = new vis.Network(container_res, data_res, options);
        }
    } else if (div_id === "study_results_popup") {
        var container_res_popup = document.getElementById("study_results_popup");
        if (js_results_data['study_results'].length === 0) {
            container_res_popup.innerHTML = "<p style='display: flex; justify-content: center; align-items: center; height: 100%;'>No study results found</p>";
        } 
        else {
            options.nodes.font.size = 14;
            network_results_popup = new vis.Network(container_res_popup, data_res, options);
        }
    } else if (div_id === "collective_visualization") {
        var container = document.getElementById("collective_visualization");
        if (js_results_data['study_results'].length === 0) {
            container.innerHTML = "<p style='display: flex; justify-content: center; align-items: center; height: 100%;'>No study results found</p>";
        } else {
            options.nodes.font.size = 14;
            network_results = new vis.Network(container, data_res, options);
        }
    }
}

var modal_res = document.getElementById("myModal_result");

// Get the button that opens the modal
var btn_res = document.getElementById("myBtn_result");

// Get the <span> element that closes the modal
var span_res = document.getElementsByClassName("close")[2];

if (btn_res) {
    btn_res.onclick = function() {
        modal_res.style.display = "block";
    }
}

// When the user clicks on <span> (x), close the modal
span_res.onclick = function () {
    modal_res.style.display = "none";
}

var modal_collective = document.getElementById("myModal_collective");
var btn_collective = document.getElementById("myBtn_collective");
var span_collective = document.getElementsByClassName("close")[3];

btn_collective.onclick = function() {
    modal_collective.style.display = "block";
}

span_collective.onclick = function() {
    modal_collective.style.display = "none";
}

// window.addEventListener("load", () => {
//   cache_obs_outcome();
//   draw_study_result();
// });

