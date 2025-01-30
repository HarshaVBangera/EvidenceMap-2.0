$(".collapseHeader").click(function () {
    $collapseHeader = $(this);
    //getting the next element
    $collapseContent = $collapseHeader.next();
    //open up the content needed - toggle the slide- if visible, slide up, if not slidedown.
    $collapseContent.slideToggle(500, function () {
    });

});


function generateCollectiveMap() {
    const container = document.getElementById('collective_visualization');
    container.style.display = 'block';
    
    container.innerHTML = '<div class="loading">Loading visualization...</div>';
    fetch('/visualize/collective')
        .then(response => response.json())
        .then(data => {
            console.log('Received data:', data);
            if (data.study_results && data.study_results.length > 0) {
                drawCollectiveVisualization(data.study_results, data.colors);
            } else {
                container.innerHTML = 'No study results available for visualization';
            }
        })
        .catch(error => {
            console.error('Error:', error);
            container.innerHTML = 'Error generating visualization';
        });
}

function openFullScreen() {
    const visualizationWindow = window.open('', '_blank');
    visualizationWindow.document.write(`
        <html>
            <head>
                <title>Full Screen Evidence Map</title>
                <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <link href="https://unpkg.com/vis-network/styles/vis-network.min.css" rel="stylesheet" type="text/css">
                <style>
                    body, html { 
                        margin: 0; 
                        padding: 0; 
                        height: 100%; 
                        width: 100%; 
                    }
                    #fullscreen-visualization {
                        height: 100vh;
                        width: 100vw;
                    }
                </style>
            </head>
            <body>
                <div id="fullscreen-visualization"></div>
            </body>
        </html>
    `);
    
    // Re-create the visualization in the new window
    fetch('/visualize/collective')
        .then(response => response.json())
        .then(data => {
            drawCollectiveVisualization(data.study_results, data.colors, visualizationWindow.document.getElementById('fullscreen-visualization'));
        });
}

function drawCollectiveVisualization(studyResults, colors, customContainer = null) {
    const container = customContainer || document.getElementById('collective_visualization');
    console.log('Study Results:', studyResults); 
    
    const nodes = new vis.DataSet();
    const edges = new vis.DataSet();
    const nodeMap = new Map();

    studyResults.forEach((result) => {
        if (result.label) {  // Ensure label exists
            console.log('Creating node:', result.label, 'with type:', result.type);
            nodeMap.set(result.label, {
                id: result.label,
                label: result.term || result.label,
                type: result.type,
                color: colors[result.type] || '#d3d3d3',
                shape: 'box',
                font: { size: 14, bold: true },
                borderWidth: 2
            });
        }
    });
        

    nodes.add(Array.from(nodeMap.values()));
    
    console.log('Available nodes:', Array.from(nodeMap.keys()));

    studyResults.forEach((result) => {
        if (result.connects_to && Array.isArray(result.connects_to)) {
            console.log(`Processing connections for ${result.label}:`, result.connects_to);
            result.connects_to.forEach(targetLabel => {
                const cleanTargetLabel = targetLabel.trim();
                if (nodeMap.has(cleanTargetLabel)) {
                    console.log('Valid edge:', result.label, '->', cleanTargetLabel);
                    edges.add({
                        from: result.label,
                        to: cleanTargetLabel,
                        arrows: 'to',
                        width: 2,
                        color: { color: '#848484', opacity: 0.8 }
                    });
                } else {
                    console.log('Missing target node:', cleanTargetLabel);
                }
            });
        }
    });


    const network = new vis.Network(container, 
    { nodes, edges }, 
    {
        physics: {
            enabled: true,
            stabilization: {
                iterations: 500,
                updateInterval: 50
            },
            barnesHut: {
                springLength: 400,
                damping: 0.09,
                avoidOverlap: 1
            }
        },
        nodes: {
            margin: 15,
            widthConstraint: {
                minimum: 100,
                maximum: 250
            },
            font: {
                size: 16,
                face: 'arial',
                bold: true
            },
            borderWidth: 2,
            shadow: true
        },
        edges: {
            smooth: {
                type: 'dynamic',
                
            },
            width: 2,
            arrows: {
                to: {
                    enabled: true,
                    scaleFactor: 0.7
                }
            }
        },
        layout: {
            improvedLayout: true,
            hierarchical: {
                direction: 'LR',
                sortMethod: 'directed',
                levelSeparation: 250,
                nodeSpacing: 500,
                treeSpacing: 250,
                blockShifting: true,
                edgeMinimization: true
            }
        },
        interaction: {
            dragNodes: true,
            dragView: true,
            zoomView: true,
            hover: true,
            navigationButtons: true,
            keyboard: true
        },
        manipulation: {
            enabled: true
        }
    }
);


}


// Define the color palette for the visualization
const colorPalette = {
    'Intervention': '#FF5733',
    'Observation': '#33FF57',
    'Count': '#3357FF',
    'Outcome': '#FFC300'
};

// Add a listener to the DOMContentLoaded event
document.addEventListener('DOMContentLoaded', function () {
    generateCollectiveMap(); // Generate the collective map when the document is loaded
    
});

        