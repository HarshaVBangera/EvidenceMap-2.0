var nodes_enroll = null;
var network_enroll = null;
var network_enroll_popup = null;

function draw_design(js_enroll_data, div_id, colors) {
    //load enrollment data
    var included_ls = [];
    var excluded_ls = [];
    for (var i = 0; i < js_enroll_data.length; i++) {
        // split by negated
        if (js_enroll_data[i]['negation'] === "affirmed" || js_enroll_data[i]['negation'] === true) {
            //excluded
            excluded_ls.push(js_enroll_data[i]['term']);
        } else if (js_enroll_data[i]['negation'] === "negated" || js_enroll_data[i]['negation'] === false) {
            //includes
            included_ls.push(js_enroll_data[i]['term']);
        }
    }

    // add nodes
    var in_num = included_ls.length;
    var ex_num = excluded_ls.length;
    var x = 0;
    var x_step = 1000 / (in_num + ex_num + 2) + 10
    nodes_enroll = [];

    // add text "included"
    if (in_num > 0) {
        nodes_enroll.push({
            id: 1000,
            label: '<b>Included</b>',
            shape: "text",
            x: (2 * x + x_step * (in_num + 1)) / 2,
            y: 150,
            font: {
                multi: 'html',
                bold: '18px arial black'
            },
            widthConstraint: 100,
        })
    }


    // add included nodes
    for (var j = 0; j < in_num; j++) {
        let id = j;
        x = x + x_step;
        nodes_enroll.push({
            id: id,
            label: included_ls[j],
            x: x,
            y: 200,
            group: 'Included',
            color: colors['Participant'],
            widthConstraint: 100,
        })
    }


    if (in_num > 0) {
        x = x + x_step; // add a blank between the includes nodes and excluded nodes
    }

    // add text "excluded"
    if (ex_num > 0) {
        nodes_enroll.push({
            id: 2000,
            label: '<b>Excluded</b>',
            shape: "text",
            //x: (2*x+x_step*(2*in_num+ex_num+3))/2,
            x: (2 * x + x_step * (ex_num + 1)) / 2,
            y: 150,
            font: {
                multi: 'html',
                bold: '18px arial black'
            },
            widthConstraint: 100,
        })
    }
    // add excluded nodes
    for (var j = 0; j < ex_num; j++) {
        let id = j + in_num + 1;
        x = x + x_step;
        nodes_enroll.push({
            id: id,
            label: excluded_ls[j],
            x: x,
            y: 200,
            group: 'Included',
            color: colors['Participant'],
            widthConstraint: 100,
        })
    }


    // add edges
    var edges_enroll = [];

    // create container and network
    var data_enroll = {
        nodes: nodes_enroll,
        edges: edges_enroll,
    };
    var options = {
        // width: '100%',
        // autoResize: true,
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
                size: 16,
            },
            margin: 10,
            widthConstraint: {
                maximum: 150,
            },
        },
        physics: {
            enabled: false,
        },
    };

    if (div_id === "enrollment") {
        var container_enroll = document.getElementById("enrollment");
        network_enroll = new vis.Network(container_enroll, data_enroll, options);
    } else if (div_id === "enrollment_popup") {
        var container_enroll_popup = document.getElementById("enrollment_popup");
        options.nodes.font.size = 16;
        network_enroll_popup = new vis.Network(container_enroll_popup, data_enroll, options);
    }
}

// Get the modal
var modal = document.getElementById("myModal");

// Get the button that opens the modal
var btn = document.getElementById("myBtn");

// Get the <span> element that closes the modal
var span = document.getElementsByClassName("close")[0];

btn.onclick = function () {
    modal.style.display = "block";
}

// When the user clicks on <span> (x), close the modal
span.onclick = function () {
    modal.style.display = "none";
}

// When the user clicks anywhere outside of the modal, close it
window.onclick = function (event) {
    if (event.target == modal) {
        modal.style.display = "none";
    }
}


