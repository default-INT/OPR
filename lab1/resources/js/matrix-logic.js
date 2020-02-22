function generateMatrix() {
    let matrixSizeEntry = document.getElementById("matrixSize");
    let size = parseInt(matrixSizeEntry.value);
    if (!!size && size >= 5 && size <= 10) {
        let tableBody = document.getElementById("tableBody");
        tableBody.innerHTML = "";
        for (let i = 0; i < size; i++) {
            let tr = element("tr");
            for (let j = 0; j < size; j++) {
                let td = element("td");
                td.appendChild(element("input"));
                tr.appendChild(td);
            }

            tableBody.appendChild(tr);
        }
    } else if (!!size) {

        matrixSizeEntry.value = size > 10 ? 10 : size < 5 ? 5 : matrixSizeEntry.value;
    }
}

function calculateTimeTransp(option) {
    let line = 0;
    let numArr = [[]];
    //let size = parseInt(document.getElementById("matrixSize").value);
    let inputs = document.querySelectorAll("#tableBody input");
    let size = Math.sqrt(inputs.length);
    inputs.forEach((input, index) => {
        input.style.background = "#E6E6E6";
        input.style.color = "black";
        if (index % size === 0 && index !== 0) {
            line++;
            numArr.push([])
        }
        let num = parseInt(input.value);
        if (num !== 0 && !num) {
            return;
        }
        numArr[line].push(num)
    });
    let columns = [0];
    let sum = 0;
    let outStr = "";
    for (let i = 0; i < size; i++) {
        if (option === 'max') {
            let values = getMaxValues(columns, numArr);
            let item = size * (values.rowMax + 1) - (size - (values.colMax + 1));

            inputs[item - 1].style.background = "red";
            inputs[item - 1].style.color = "white";

            outStr += "P" + (values.rowMax + 1) + " - (" + values.max + ") - ";
            if (i + 1 === size) outStr += "P" + (values.colMax + 1);
            columns.push(values.colMax);
            sum += values.max;
        } else if (option === 'min') {
            let values = getMinValues(columns, numArr);
            let item = size * (values.rowMin + 1) - (size - (values.colMin + 1));

            inputs[item - 1].style.background = "#f668fb";
            inputs[item - 1].style.color = "white";

            outStr += "P" + (values.rowMin + 1) + " - (" + values.min + ") - ";
            if (i + 1 === size) outStr += "P" + (values.colMin + 1);
            columns.push(values.colMin);
            sum += values.min;
        } else if (option === 'exMin') {
            let values = exhaustiveSearchMin(numArr, i);
            let item = size * (values.rowMin + 1) - (size - (values.colMin + 1));

            inputs[item - 1].style.background = "#f668fb";
            inputs[item - 1].style.color = "white";

            outStr += "P" + (values.rowMin + 1) + " - (" + values.min + ") - ";
            if (i + 1 === size) outStr += "P" + (values.colMin + 1);
            columns.push(values.colMin);
            sum += values.min;
        } else if (option === 'exMax') {
            let values = exhaustiveSearchMax(numArr, i);
            let item = size * (values.rowMax + 1) - (size - (values.colMax + 1));

            inputs[item - 1].style.background = "red";
            inputs[item - 1].style.color = "white";

            outStr += "P" + (values.rowMax + 1) + " - (" + values.max + ") - ";
            if (i + 1 === size) outStr += "P" + (values.colMax + 1);
            columns.push(values.colMax);
            sum += values.max;
        }

    }
    outStr += " = " + sum + " ед.";
    document.getElementById("out").innerHTML = outStr;
}

function exhaustiveSearchMin(numArr, i) {
    let rowMin;
    let colMin;
    let min;
    if ((i + 1) !== numArr.length) {
        min = numArr[i][i + 1];
        rowMin = i;
        colMin = i + 1;
        numArr[i].forEach((el, index) => {
            if (index > i && el < min) {
                min =  el;
                colMin = index;
            }
        });
    } else {
        min = numArr[i][0];
        rowMin = i;
        colMin = 0;
    }
    return {min, rowMin, colMin}
}

function exhaustiveSearchMax(numArr, i) {
    let rowMax;
    let colMin;
    let max;
    if ((i + 1) !== numArr.length) {
        max = numArr[i][i + 1];
        rowMax = i;
        colMax = i + 1;
        numArr[i].forEach((el, index) => {
            if (index > i && el > max) {
                max =  el;
                colMax = index;
            }
        });
    } else {
        max = numArr[i][0];
        rowMax = i;
        colMax = 0;
    }
    return {max, rowMax, colMax}
}

function getMinValues(columns, numArr) {
    let size = numArr.length;
    let min = undefined;
    let colMin = 0;
    let rowMin = 0;
    let lastCol = columns[columns.length - 1];
    if (columns.length !== size) {
        for (let j = 0; j < size; j++) {
            if (j !== lastCol) {
                let check = false;
                columns.forEach(el => {
                    if (j === el) check = true;
                });
                if (!check && (numArr[lastCol][j] < min || min === undefined)) {
                    min = numArr[lastCol][j];
                    colMin = j;
                    rowMin = lastCol;
                }
            }
        }
    } else {
        min = numArr[lastCol][columns[0]];
        colMin = columns[0];
        rowMin = lastCol;
    }
    return {min, colMin, rowMin}
}

function getMaxValues(columns, numArr) {
    let size = numArr.length;
    let max = undefined;
    let colMax = 0;
    let rowMax = 0;
    let lastCol = columns[columns.length - 1];
    if (columns.length !== size) {
        for (let j = 0; j < size; j++) {
            if (j !== lastCol) {
                let check = false;
                columns.forEach(el => {
                    if (j === el) check = true;
                });
                if (!check && (numArr[lastCol][j] > max || max === undefined)) {
                    max = numArr[lastCol][j];
                    colMax = j;
                    rowMax = lastCol;
                }
            }
        }
    } else {
        max = numArr[lastCol][columns[0]];
        colMax = columns[0];
        rowMax = lastCol;
    }
    return {max, colMax, rowMax}
}