/*function loadData() {
    let request = new XMLHttpRequest();
    request.open("GET", "http://127.0.0.1:5000/");
    request.responseType = 'json';

    request.onload = function() {
        buildVisualization(request.response)
    };
    request.send()
}*/


function loadData(data) {
    buildVisualization(data)
    //let request = new XMLHttpRequest();
    //request.open("GET", "http://127.0.0.1:5000/");
    //request.responseType = 'json';
    //request.onload = function() {
    //   buildVisualization(data)
    //};
    //request.send()
}

function initializeElt(eltTag, innerHTML) {
    // Clean way to generate an HTML element that is of a specific tag-type
    newNode = document.createElement(eltTag)
    newNode.innerHTML = innerHTML
    return newNode
}

function buildVisualization(data) {
    // Main driver function, calls subfunctions
    const fmtData = reformatData(data);
    console.log(fmtData)

    buildOverview(fmtData)
    buildSentences(fmtData["sentences"])
}

function buildOverview(data) {
    const overviewDiv = document.getElementById("overview")

    // Add Document ID
    var doc_link = "https://pubmed.ncbi.nlm.nih.gov/".concat(data["doc_id"])
    // overviewDiv.appendChild(initializeElt("p", "<b>Document ID</b>: " + data["doc_id"].link(doc_link)))
    // //overviewDiv.appendChild(initializeElt("p", "<b>Question Type</b>: " + data["type of study"]))
    // overviewDiv.appendChild(initializeElt("p", "<b>Title</b>: "  + data["title"].italics()))
    //overviewDiv.appendChild(initializeElt("p", "<b>Abstract</b>: " + data["abstract"]))//
}

function mergeCell(table1, startRow, endRow, col) {
    var tb = document.getElementById(table1);
    if (!tb || !tb.rows || tb.rows.length <= 0) {
        return;
    }
    if (col >= tb.rows[0].cells.length || (startRow >= endRow && endRow != 0)) {
        return;
    }
    if (endRow == 0) {
        endRow = tb.rows.length - 1;
    }
    for (var i = startRow; i < endRow; i++) {
        if (tb.rows[startRow].cells[col].innerHTML == tb.rows[i + 1].cells[col].innerHTML) {
            tb.rows[i + 1].removeChild(tb.rows[i + 1].cells[col]);
            tb.rows[startRow].cells[col].rowSpan = (tb.rows[startRow].cells[col].rowSpan) + 1;
        } else {
            mergeCell(table1, i + 1, endRow, col);
            break;
        }
    }
}


var displacy = new displaCy('http://localhost:8080', {
    container: '#displacy',
})

function parse(text) {
    displacy.parse(text)
}


function buildSentences(sentencesData) {

    // Break down the results by sentence
    const allSentenceDiv = document.getElementById("sentences")
    var currentSection = null;
    var lastSection = null;
    var currentTextann = null;
    var currentMEP = [];
    var index = 0;
    var dataLength = sentencesData.length - 1;

    sentencesData.forEach(function (datum) {
        console.log(datum)
        if (currentSection !== datum["section"] && lastSection != "CONCLUSIONS") {

            //sentenceDiv = initializeElt('div', "")

            if (index > 0) {
                // last section header
                const newHeader = initializeElt("h3", currentSection)
                newHeader.setAttribute('class', 'sentence-section-header')
                allSentenceDiv.appendChild(newHeader)

                //last section text with PICO ann
                sentenceDiv = initializeElt('div', "")
                sentenceDiv.setAttribute('class', 'sentence-block content')
                // sentenceDiv.innerHTML+='<b>Text</b>:' + currentTextann
                sentenceDiv.innerHTML += currentTextann
                //last section MEP
                if (currentMEP.length != 0) {

                    sentenceDiv.appendChild(initializeElt('h3', "<div style='margin-top:20px;font-size:20px'><b>Evidence Propositions</b></div>"))
                    MepTable = initializeElt('table', '')
                    MepTableHeader = initializeElt('tr', '')
                    MepTableHeader.appendChild(initializeElt('th', 'Intervention'))
                    MepTableHeader.appendChild(initializeElt('th', 'Outcome'))
                    MepTableHeader.appendChild(initializeElt('th', 'Observation'))

                    MepTableHeader.appendChild(initializeElt('th', 'Count'))
                    MepTableHeader.appendChild(initializeElt('th', 'negation'))
                    MepTable.appendChild(MepTableHeader)
                    currentMEP.forEach(function (d) {
                        d.forEach(function (datum2) {

                            MepTableRow = initializeElt('tr', '')

                            MepTableRow.appendChild(initializeElt('td', datum2['Intervention']))
                            MepTableRow.appendChild(initializeElt('td', datum2['Outcome']))
                            MepTableRow.appendChild(initializeElt('td', datum2['Observation']))

                            MepTableRow.appendChild(initializeElt('td', datum2['Count']))
                            MepTableRow.appendChild(initializeElt('td', datum2['negation']))
                            MepTable.appendChild(MepTableRow)
                        })
                    })

                    sentenceDiv.appendChild(MepTable)
                }
                allSentenceDiv.appendChild(sentenceDiv)

                currentTextann = null
                currentMEP = []
            }
            if (lastSection != "CONCLUSIONS") {
                lastSection = currentSection
            }
            currentSection = datum["section"]
        }


        // text annotation with PICO
        if (currentTextann == null) {
            currentTextann = datum["text_ann"]
        } else {
            currentTextann += datum["text_ann"]
        }

        // Evidence Propositions Tables
        if (datum["Evidence Propositions"].length != 0) {
            currentMEP.push(datum["Evidence Propositions"])
        }
        index++

        // check if it's the last sentence
        if (index > dataLength) {

            // last section header
            if (currentSection !== lastSection && lastSection != "CONCLUSIONS") {
                const newHeader = initializeElt("h3", currentSection)
                newHeader.setAttribute('class', 'sentence-section-header')
                allSentenceDiv.appendChild(newHeader)
            }

            //last section text with PICO ann
            sentenceDiv = initializeElt('div', "")
            sentenceDiv.setAttribute('class', 'sentence-block content')
            sentenceDiv.innerHTML += currentTextann

            //last section MEP
            if (currentMEP.length != 0) {

                sentenceDiv.appendChild(initializeElt('h3', "<div style='margin-top:20px;font-size:20px'><b>Evidence Propositions</b></div>"))
                MepTable = initializeElt('table', '')
                MepTableHeader = initializeElt('tr', '')
                MepTableHeader.appendChild(initializeElt('th', 'Intervention'))
                MepTableHeader.appendChild(initializeElt('th', 'Outcome'))
                MepTableHeader.appendChild(initializeElt('th', 'Observation'))

                MepTableHeader.appendChild(initializeElt('th', 'Count'))
                MepTableHeader.appendChild(initializeElt('th', 'negation'))
                MepTable.appendChild(MepTableHeader)
                currentMEP.forEach(function (d) {
                    d.forEach(function (datum2) {

                        MepTableRow = initializeElt('tr', '')

                        MepTableRow.appendChild(initializeElt('td', datum2['Intervention']))
                        MepTableRow.appendChild(initializeElt('td', datum2['Outcome']))
                        MepTableRow.appendChild(initializeElt('td', datum2['Observation']))

                        MepTableRow.appendChild(initializeElt('td', datum2['Count']))
                        MepTableRow.appendChild(initializeElt('td', datum2['negation']))
                        MepTable.appendChild(MepTableRow)
                    })
                })

                sentenceDiv.appendChild(MepTable)
            }
            allSentenceDiv.appendChild(sentenceDiv)


        }


    });
}

function buildPopulation(PopulationData) {
    const designDiv = document.getElementById("population")
    designDiv.appendChild(initializeElt('h3', "Enrollement"))
    const partTable = initializeElt('table', "")
    const partTableHeader = initializeElt('tr', "")
    partTableHeader.appendChild(initializeElt('th', 'Term'))
    partTableHeader.appendChild(initializeElt('th', 'Negation'))
    partTableHeader.appendChild(initializeElt('th', 'UMLS CUI'))
    partTable.appendChild(partTableHeader)
    var pico_keys = ["Participant", "Intervention", "Outcome"]
    for (const key in pico_keys) {
        //designDiv.appendChild(initializeElt('h3', "Participants"))

        designData[pico_keys[key]].forEach(function (datum) {
            const partTableRow = initializeElt('tr', "")
            partTableRow.appendChild(initializeElt('td', datum["term"]))
            partTableRow.appendChild(initializeElt('td', datum["negation"]))
            partTableRow.appendChild(initializeElt('td', datum["umls"]))
            partTable.appendChild(partTableRow)
        })
        designDiv.appendChild(partTable)
    }
}

function buildStudyDesign(designData) {
    const designDiv = document.getElementById("design")

    // Hypothesis information
    designDiv.appendChild(initializeElt('h3', "Hypothesis"))
    const hypTable = initializeElt('table', "")

    // Build table header
    const hypTableHeader = initializeElt('tr', "")
    hypTableHeader.appendChild(initializeElt('th', 'Intervention'))
    hypTableHeader.appendChild(initializeElt('th', 'Observation'))
    hypTableHeader.appendChild(initializeElt('th', 'Outcome'))

    hypTable.appendChild(hypTableHeader)
    designData["Hypothesis"].forEach(function (datum) {
        // Append the Intervention/Observation/Outcome rows for the hypothesis
        const hypTableRow = initializeElt('tr', "")
        hypTableRow.appendChild(initializeElt('td', datum["Intervention"]))
        hypTableRow.appendChild(initializeElt('td', datum["Observation"]))
        hypTableRow.appendChild(initializeElt('td', datum["Outcome"]))
        hypTable.appendChild(hypTableRow)
    })
    designDiv.appendChild(hypTable)

    designDiv.appendChild(initializeElt('h3', "PICO Elements"))
    const partTable = initializeElt('table', "")
    const partTableHeader = initializeElt('tr', "")
    partTableHeader.appendChild(initializeElt('th', 'PICO'))
    partTableHeader.appendChild(initializeElt('th', 'Term'))
    partTableHeader.appendChild(initializeElt('th', 'Negation'))
    partTableHeader.appendChild(initializeElt('th', 'UMLS CUI'))
    partTable.appendChild(partTableHeader)
    var pico_keys = ["Participant", "Intervention", "Outcome"]
    for (const key in pico_keys) {
        //designDiv.appendChild(initializeElt('h3', "Participants"))

        designData[pico_keys[key]].forEach(function (datum) {
            const partTableRow = initializeElt('tr', "")
            partTableRow.appendChild(initializeElt('td', pico_keys[key]))
            partTableRow.appendChild(initializeElt('td', datum["term"]))
            partTableRow.appendChild(initializeElt('td', datum["negation"]))
            partTableRow.appendChild(initializeElt('td', datum["umls"]))
            partTable.appendChild(partTableRow)
        })
        designDiv.appendChild(partTable)
    }

}

function buildStudyResults(resultsData) {
    const designDiv = document.getElementById("results")


    // StudyArm 1
    // Same logic as Hypothesis
    designDiv.appendChild(initializeElt('h3', "Study Arm 1: ".concat(resultsData["Arm 1"]["term"].italics()))) // resultsData["Arm 1"]["term"]
    const intTable = initializeElt('table', "")
    const intTableHeader = initializeElt('tr', "")
    intTableHeader.appendChild(initializeElt('th', 'Outcome'))
    intTableHeader.appendChild(initializeElt('th', 'Observation'))
    intTableHeader.appendChild(initializeElt('th', 'Count'))
    intTable.appendChild(intTableHeader)
    resultsData["Arm 1"]["results"].forEach(function (datum) {
        const intTableRow = initializeElt('tr', "")
        intTableRow.appendChild(initializeElt('td', datum["Outcome"]))
        intTableRow.appendChild(initializeElt('td', datum["Observation"]))
        intTableRow.appendChild(initializeElt('td', datum["Count"]))
        intTable.appendChild(intTableRow)
    })
    designDiv.appendChild(intTable)

    // Study Arm 2
    // Same logic as Hypothesis
    designDiv.appendChild(initializeElt('h3', "Study Arm 2: ".concat(resultsData["Arm 2"]["term"].italics())))

    const outTable = initializeElt('table', "")
    const outTableHeader = initializeElt('tr', "")
    outTableHeader.appendChild(initializeElt('th', 'Outcome'))
    outTableHeader.appendChild(initializeElt('th', 'Observation'))
    outTableHeader.appendChild(initializeElt('th', 'Count'))
    outTable.appendChild(outTableHeader)
    resultsData["Arm 2"]["results"].forEach(function (datum) {
        const outTableRow = initializeElt('tr', "")
        outTableRow.appendChild(initializeElt('td', datum["Outcome"]))
        outTableRow.appendChild(initializeElt('td', datum["Observation"]))
        outTableRow.appendChild(initializeElt('td', datum["Count"]))
        outTable.appendChild(outTableRow)
    })
    designDiv.appendChild(outTable)

    // Hypothesis information
    designDiv.appendChild(initializeElt('h3', "All Arms"))
    const hypTable = initializeElt('table', "")

    // Build table header
    const hypTableHeader = initializeElt('tr', "")

    hypTableHeader.appendChild(initializeElt('th', 'Outcome'))
    hypTableHeader.appendChild(initializeElt('th', 'Observation'))
    hypTableHeader.appendChild(initializeElt('th', 'Count'))

    hypTable.appendChild(hypTableHeader)
    resultsData["All Arms"].forEach(function (datum) {
        // Append the Intervention/Observation/Outcome rows for the hypothesis
        const hypTableRow = initializeElt('tr', "")

        hypTableRow.appendChild(initializeElt('td', datum["Outcome"]))
        hypTableRow.appendChild(initializeElt('td', datum["Observation"]))
        hypTableRow.appendChild(initializeElt('td', datum["Count"]))
        hypTable.appendChild(hypTableRow)
    })
    designDiv.appendChild(hypTable)

    // Participants
    // Same logic as Hypothesis
    designDiv.appendChild(initializeElt('h3', "Comparison"))
    const partTable = initializeElt('table', "")
    const partTableHeader = initializeElt('tr', "")
    partTableHeader.appendChild(initializeElt('th', 'Intervention'))
    partTableHeader.appendChild(initializeElt('th', 'Observation'))
    partTableHeader.appendChild(initializeElt('th', 'Outcome'))
    partTableHeader.appendChild(initializeElt('th', 'Count'))


    partTable.appendChild(partTableHeader)
    resultsData["Comparison"].forEach(function (datum) {
        const partTableRow = initializeElt('tr', "")
        partTableRow.appendChild(initializeElt('td', datum["Intervention"]))
        partTableRow.appendChild(initializeElt('td', datum["Observation"]))
        partTableRow.appendChild(initializeElt('td', datum["Outcome"]))
        partTableRow.appendChild(initializeElt('td', datum["Count"]))
        partTable.appendChild(partTableRow)
    })
    designDiv.appendChild(partTable)
}

function reformatData(data) {
    // Any post-processing of the data goes here
    return data;
}