var nodes_res = null;
var edges_res = null;
var network_results = null;
var network_results_popup = null;

//var js_results_data = js_data["study results"]
//var I_len = js_data["study design"]["Intervention"].length // use the len in study design
var I_len = null;
var Obs_len_res = null;
var Outc_len_res = null;
//var all_obs_set = null;
//var all_outcome_set = null;
//var all_obs_ls = null;
//var all_outcome_ls = null;
//var obs_count_dict = null;
//var outcome_count_dict = null;

var color_ls = ["#f6dfcb", "#eaacb9", "#bd68b0", "#6172ae", "#2c4478",
    "#f2de4d", "#f6f171", "#cee398", "#8fb4ae", "#577590"];

function cache_obs_outcome(js_results_data, num_arm) {
    //all_obs_set = new Set();
    //all_outcome_set = new Set();
    I_len = num_arm;
    //I_len = js_data["study design"]["Intervention"].length // use the len in study design
    var all_obs_ls = [];
    var all_outcome_ls = [];
    for (var i = 1; i < I_len + 1; i++) {
        let dict_k = "Arm " + i;
        //cache obs
        for (var j = 0; j < js_results_data[dict_k]["results"].length; j++) {
            all_obs_ls.push(js_results_data[dict_k]["results"][j]["Observation"]);
        }
        //cache outcome
        for (var j = 0; j < js_results_data[dict_k]["results"].length; j++) {
            all_outcome_ls.push(js_results_data[dict_k]["results"][j]["Outcome"]);
        }
    }
    //cache for comparision
    for (var j = 0; j < js_results_data["Comparison"].length; j++) {
        // consider negation: same obs, different negation: 2 unique obs
        var obs_value = js_results_data["Comparison"][j]["Observation"];
        let neg_flag = js_results_data["Comparison"][j]["negation"];
        if (neg_flag === 'negated' || neg_flag === true) {
            obs_value = 'no/not ' + obs_value;
        }
        all_obs_ls.push(obs_value); //js_results_data["Comparison"][j]["Observation"]
        all_outcome_ls.push(js_results_data["Comparison"][j]["Outcome"]);
    }

    // generate dict for count
    var obs_count_dict = Array.from(new Set(all_obs_ls)).map(a =>
        ({name: a, count: all_obs_ls.filter(f => f === a).length}));
    var outcome_count_dict = Array.from(new Set(all_outcome_ls)).map(a =>
        ({name: a, count: all_outcome_ls.filter(f => f === a).length}));

    Obs_len_res = obs_count_dict.length; //all_obs_set.size;
    Outc_len_res = outcome_count_dict.length; //all_outcome_set.size;
}


function draw_study_result(js_results_data, div_id) {

    nodes_res = [];
    edges_res = [];

    var obs_x = 0;
    var outcome_x = 0;

    var obs_step = 1000 / (Obs_len_res + 1) + 10;
    var outcome_step = 1000 / (Outc_len_res + 1) + 10;

    /// add Intervention nodes, id starts from 200
    for (var i = 0; i < I_len; i++) {
        let id = i + 200;
        let x = 0; // from -500 to 500
        let y = 200;
        let x_step = 1000 / (I_len + 1) + 10;

        let arm_id = i + 1;
        let dict_k = "Arm " + arm_id;

        nodes_res.push({
            id: id,
            label: js_results_data[dict_k]['term'],
            x: x + x_step * (i + 1),
            y: y,
            group: 'Intervention',
            color: '#9FE2BF', //"#109618", // green
            widthConstraint: 150,
        })

        // /// add "Arm X" text box
        // nodes_res.push({
        //   label: dict_k,
        //   shape: 'text', //'circle'
        //   x: x + x_step*(i+1),
        //   y: y-100,
        //   font:'18px arial black bolded',
        //   //color: "#EEEDE7",
        //   widthConstraint: {
        //     maximum: 150,
        //   },
        // });
        // nodes_res.push({
        //   id: id+1000,
        //   shape: 'dot', //'text', //'circle'
        //   size: 15,
        //   x: x + x_step*(i+1),
        //   y: y-70,
        //   color: "#EEEDE7",
        //   widthConstraint: {
        //     maximum: 150,
        //   },
        // });

        /// add "Arm X" text box
        if (i == 0) {
            nodes_res.push({
                id: id + 1000,
                label: '<b>Intervention</b>',
                shape: 'text', //'circle'
                font: {
                    multi: 'html',
                    bold: '18px arial black'
                },
                x: x + x_step * (i + 1),
                y: y - 60,
            });
        } else {
            nodes_res.push({
                id: id + 1000,
                label: '<b>Comparator</b>',
                shape: 'text', //'circle'
                font: {
                    multi: 'html',
                    bold: '18px arial black'
                },
                x: x + x_step * (i + 1),
                y: y - 60,
            });
        }

        //edges_res.push({from: id, to: id+1000, color: 'gray', arrows: {to: false}});
    }


    // Comparison: we assume the values of "Intervention" and "Comparator" is the same as the interventions in the two arms
    for (var i = 0; i < js_results_data["Comparison"].length; i++) {
        // add nodes

        var obs_value = js_results_data["Comparison"][i]["Observation"];
        var outcome_value = js_results_data["Comparison"][i]["Outcome"];
        let neg_flag = js_results_data["Comparison"][i]["negation"];
        // check negation
        if (neg_flag === 'negated' || neg_flag === true) {
            obs_value = 'no/not ' + obs_value;
        }

        // check duplicate
        let obs_dup_check = nodes_res.filter(obj => {
            return obj.label === obs_value
        });
        let outcome_dup_check = nodes_res.filter(obj => {
            return obj.label === outcome_value
        });

        if (obs_dup_check.length < 1) {
            // no duplicate nodes before
            var obs_id = i + 300;
            obs_x = obs_x + obs_step;
            nodes_res.push({
                id: obs_id,
                label: obs_value,
                x: obs_x,
                y: 300,
                color: "#98c7f3",
            });
        } else {
            // nodes existed
            // keep the id for edges
            var obs_id = obs_dup_check[0]['id']
        }


        if (outcome_dup_check.length < 1) {
            var outcome_id = i + 400;
            outcome_x = outcome_x + outcome_step;
            nodes_res.push({
                id: outcome_id,
                label: js_results_data["Comparison"][i]["Outcome"],
                x: outcome_x,
                y: 400,
                color: '#F7DC6F',
            });
        } else {
            var outcome_id = outcome_dup_check[0]['id']
        }

        // add edges
        edges_res.push({from: outcome_id, to: obs_id, color: 'purple'});
        for (var j = 0; j < I_len; j++) {
            edges_res.push({from: obs_id, to: j + 200, color: 'purple'})
        }
    }
    //var ei = 0;
    // Single Arm result
    for (var i = 0; i < I_len; i++) {
        let arm_id = i + 1;
        let dict_k = "Arm " + arm_id;

        // add obs and outcome nodes and edges, regardless of duplicate
        for (var j = 0; j < js_results_data[dict_k]["results"].length; j++) {

            let obs_value = js_results_data[dict_k]["results"][j]['Observation'];
            let outcome_value = js_results_data[dict_k]["results"][j]['Outcome'];

            // check duplicate
            let obs_dup_check = nodes_res.filter(obj => {
                return obj.label === obs_value
            });
            let outcome_dup_check = nodes_res.filter(obj => {
                return obj.label === outcome_value
            });

            var col = 'gray' // set default edge color as gray

            if (obs_dup_check.length < 1) {
                var obs_id = j + i * 100 + 3000;
                obs_x = obs_x + obs_step;
                nodes_res.push({
                    id: obs_id,
                    label: obs_value,
                    x: obs_x,
                    y: 300,
                    color: "#98c7f3", // green
                });
            } else {
                var obs_id = obs_dup_check[0]['id'];
                //col = '#98c7f3';
            }

            if (outcome_dup_check.length < 1) {
                var outcome_id = j + i * 100 + 4000;
                outcome_x = outcome_x + outcome_step;
                nodes_res.push({
                    id: outcome_id,
                    label: outcome_value,
                    x: outcome_x,
                    y: 400,
                    color: '#F7DC6F', // green
                });
            } else {
                var outcome_id = outcome_dup_check[0]['id'];
                //col = '#98c7f3';
            }

            edges_res.push({from: obs_id, to: i + 200, color: col});

            let neg_flag = js_results_data[dict_k]["results"][j]['negation'];
            if (neg_flag === true || neg_flag === "affirmed") {
                edges_res.push({from: outcome_id, to: obs_id, color: col, label: 'negated'});
            } else {
                //neg_flag === false || neg_flag === "negated"
                edges_res.push({from: outcome_id, to: obs_id, color: col});
            }
        }
    }



    // create container and network
    var data_res = {
        nodes: nodes_res,
        edges: edges_res,
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
        network_results = new vis.Network(container_res, data_res, options);
    } else if (div_id === "study_results_popup") {
        var container_res_popup = document.getElementById("study_results_popup");
        options.nodes.font.size = 14;
        network_results_popup = new vis.Network(container_res_popup, data_res, options);
    }

}

var modal_res = document.getElementById("myModal_result");

// Get the button that opens the modal
var btn_res = document.getElementById("myBtn_result");

// Get the <span> element that closes the modal
var span_res = document.getElementsByClassName("close")[2];

btn_res.onclick = function () {
    modal_res.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
span_res.onclick = function () {
    modal_res.style.display = "none";
}

// window.addEventListener("load", () => {
//   cache_obs_outcome();
//   draw_study_result();
// });

