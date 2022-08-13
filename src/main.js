import './style.css'
import { renderNetwork } from './NNRenderer'
const app = document.querySelector('#app')


class layer {
  constructor(length) {
    this.length = length
  }
}
class network {
  constructor(layers) {
    this.layers = layers
  }
}

app.appendChild(renderNetwork(
  new network([
    new layer(2),
    new layer(4),
    new layer(4),
    new layer(2)
  ])
))


