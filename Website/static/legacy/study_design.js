var nodes_des = null;
var edges_des = null;
var network_des = null;
var network_des_popup = null;

function draw_design(js_design_data, div_id, colors) {
    nodes_des = [];
    edges_des = [];

    var hyp_len = js_design_data["Hypothesis"].length;

    // calculate num of nodes
    var all_intv_set = new Set();
    var all_obs_set = new Set();
    var all_outcome_set = new Set();
    for (var i = 0; i < hyp_len; i++) {
        all_intv_set.add(js_design_data['Hypothesis'][i]['Intervention']);
        all_obs_set.add(js_design_data['Hypothesis'][i]['Observation']);
        all_outcome_set.add(js_design_data['Hypothesis'][i]['Outcome']);
    }

    // calculate x_step for the P, O, O elements
    var i_x = 0;
    var obs_x = 0;
    var outcome_x = 0;

    var i_step = 1000 / (all_intv_set.size + 1) + 10;
    var obs_step = 1000 / (all_obs_set.size + 1) + 10;
    var outcome_step = 1000 / (all_outcome_set.size + 1) + 10;

    // draw network
    for (var i = 0; i < hyp_len; i++) {
        let i_value = js_design_data['Hypothesis'][i]['Intervention'];
        let obs_value = js_design_data['Hypothesis'][i]['Observation'];
        let outcome_value = js_design_data['Hypothesis'][i]['Outcome'];

        //check duplicate node
        let i_dup_check = nodes_des.filter(obj => {
            return obj.label === i_value
        });
        let obs_dup_check = nodes_des.filter(obj => {
            return obj.label === obs_value
        });
        let outcome_dup_check = nodes_des.filter(obj => {
            return obj.label === outcome_value
        });

        // add Intervention node
        if (i_dup_check.length < 1) {
            var i_id = i + 200;
            i_x = i_x + i_step;
            nodes_des.push({
                id: i_id,
                label: i_value,
                x: i_x,
                y: 100,
                color: colors['Intervention'],
            });
        } else {
            var i_id = i_dup_check[0]['id'];
        }

        // add Observation node
        if (obs_dup_check.length < 1) {
            var obs_id = i + 300;
            obs_x = obs_x + obs_step;
            nodes_des.push({
                id: obs_id,
                label: obs_value,
                x: obs_x,
                y: 200,
                color: colors['Observation'],
            });
        } else {
            var obs_id = obs_dup_check[0]['id'];
        }

        // add Outcome node
        if (outcome_dup_check.length < 1) {
            var outcome_id = i + 400;
            outcome_x = outcome_x + outcome_step;
            nodes_des.push({
                id: outcome_id,
                label: outcome_value,
                x: outcome_x,
                y: 300,
                color: colors['Outcome'],
            });
        } else {
            var outcome_id = outcome_dup_check[0]['id'];
        }

        // add edge: observation to intervention
        edges_des.push({from: i_id, to: obs_id, color: 'gray'});

        // add edge: outcome to observation
        let neg_flag = js_design_data['Hypothesis'][i]['negation'];
        if (neg_flag === true || neg_flag === "affirmed") {
            edges_des.push({from: obs_id, to: outcome_id, color: 'gray', label: 'negated'});
        } else {
            //neg_flag === false || neg_flag === "negated"
            edges_des.push({from: obs_id, to: outcome_id, color: 'gray'});
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
        network_des = new vis.Network(container_des, data_des, options);
    } else if (div_id === "study_design_popup") {
        var container_des_popup = document.getElementById("study_design_popup");
        options.nodes.font.size = 14;
        network_des_popup = new vis.Network(container_des_popup, data_des, options);
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

