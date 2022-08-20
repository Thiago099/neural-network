import './style.css'

import { network } from './network/network'
import { renderNetwork } from './network/render'

const app = document.querySelector('#app')


const networkShapeElement = document.getElementById('network-shape')
const xElement = document.getElementById('x')
const yElement = document.getElementById('y') 


var input = document.getElementById('input')
var output = document.getElementById('output')

var fields = []
function model(element, defaultValue, callback)
{
  const result = document.querySelector(element)
  fields.push(result)
  result.value = localStorage.getItem(element) || defaultValue
  result.addEventListener('keydown', (e) => {
    if (e.keyCode === 13) {
      callback(result.value)
      localStorage.setItem(element, result.value)
    }
  })
  return result;
}


const uploadElement = document.getElementById('upload')

uploadElement.addEventListener('click', (e) => {
    // create file element and click
    const fileElement = document.createElement('input')
    fileElement.type = 'file'
    fileElement.click()
    // when file is selected
    fileElement.onchange = (e) => {
        // get file
        const file = fileElement.files[0]
        // create reader
        const reader = new FileReader()
        // when reader is done
        reader.addEventListener('load', (e) => {
            // get result
            const result = JSON.parse(reader.result)
            for(const item of fields) {
                item.value = result[item.id]
                localStorage.setItem(item.id, result[item.id])
            }
            myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
            x = eval(`[${xElement.value}]`)
            y = eval(`[${yElement.value}]`)
            train();
        })
        // read file
        reader.readAsText(file)
    }
})

const downloadElement = document.getElementById('download')
downloadElement.addEventListener('click', (e) => {
    const result = {}
    for(const item of fields) {
        result[item.id] = item.value
    }
    const file = new Blob([JSON.stringify(result,null,2)], {type: 'application/json'})
    const fileURL = URL.createObjectURL(file)
    const a = document.createElement('a')
    a.href = fileURL
    a.download = 'model.json'
    a.click()
}   )



model('#network-shape', '2,2', networkShape => {
  myNetwork = new network(...eval(`[${networkShape}]`))
  train()
})

model('#x', '[1,2], [2,1]', xv => {
  x = eval(`[${xv}]`)
  myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
  train()
})

model('#y', '[2,1], [1,2]', yv => {
  y = eval(`[${yv}]`)
  myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
  train()
})

model('#input', '1,2', input => {
  update()
})

var myNetwork = new network(...eval(`[${networkShapeElement.value}]`))
var x = eval(`[${xElement.value}]`)
var y = eval(`[${yElement.value}]`)

function update() {
  if(input.value == '') return
  output.value = myNetwork.Predict(eval(`[${input.value}]`)).map(x => Math.round(x))
}

const stdout = document.getElementById('stdout')
function train(){
  
  myNetwork.Learn(x,y,1000)

  stdout.innerHTML = JSON.stringify(myNetwork.layers
    .slice(1)
    .map(layer => 
      [...layer.weights.map(y=>y.map(x=>{var c = x.toFixed(2); return c == 0?0:Number(c)})),
      ...layer.bias.map(y=>{var c = y.toFixed(2); return c == 0?0:Number(c)})]), null, 2)
    .replace(/\n/g, '<br>')
    .replace(/ /g, '&nbsp;')
  
  update()
  
  app.innerHTML = "";
  app.appendChild(renderNetwork(myNetwork)) 

}
train()
// stdout

