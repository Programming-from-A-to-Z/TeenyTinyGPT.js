// Programming from A to Z
// https://github.com/Programming-from-A-to-Z/A2Z-F26
//
// TinyGPT

let filename = 'data/shakespeare.txt'; // try 'data/dickinson.txt' or your own .txt file!
let contextLength = 32; // how many tokens the model can see at once
let embedSize = 64; // the size of the each token's vector representation (the "embedding")
let batchSize = 32; // how many samples for each training step
let temperature = 0.8;

let corpus; // the training data
let vocab; // every unique token (character) in the text
let data; // the whole text as a list of character ids
let gpt; // the model!

let training = true;
let steps = 0;
let losses = [];

// Everything the model has generated, and the training step when each line began
let generated = '\n';
let lines = [{ step: 0, text: '' }];
let slider;

async function setup() {
  createCanvas(800, 590);
  textFont('monospace');

  // Load the text and build the vocabulary
  let raw = await fetch(filename);
  corpus = await raw.text();
  // A Set keeps only one copy of each character
  let unique = new Set(corpus);
  // Turn it into an array, sorted so the ids are consistent
  vocab = Array.from(unique);
  vocab.sort();
  // Encode the corpus!
  data = encode(corpus);

  // Wait for TensorFlow.js to be ready!
  await tf.ready();
  gpt = new TinyGPT(vocab.length, contextLength, embedSize);

  createButton('pause / train').mousePressed(toggleTraining).position(320, 20);
  // Speed: how many frames (training steps) per second, from 1 to 60
  createSpan('speed').position(420, 22);
  slider = createSlider(1, 60, 60).position(470, 20);
}

function toggleTraining() {
  training = !training;
}

// transform text into a list of character ids
function encode(str) {
  let ids = [];
  for (let char of str) {
    // A character's id is its index in the vocab array
    ids.push(vocab.indexOf(char));
  }
  return ids;
}

function draw() {
  background(255);
  if (!gpt) return;

  // The slider sets how many times per second draw() runs
  frameRate(slider.value());

  if (training) {
    let { inputs, targets } = getBatch();
    let loss = gpt.train(inputs, targets);
    tf.dispose([inputs, targets]);
    losses.push(loss);
    steps++;
  }
  // The model writes a character whether or not it's training
  generateCharacter();

  drawLosses(20, 40, width - 40, 140);
  drawWriting(20, 200);
  drawAttention(20, 540);
}

// Get a random block of context length text, with each target shifted by one character.
// At every position, the right answer is the next character.
function getBatch() {
  let inputs = [];
  let targets = [];
  for (let i = 0; i < batchSize; i++) {
    let start = floor(random(data.length - contextLength - 1));
    inputs.push(data.slice(start, start + contextLength));
    targets.push(data.slice(start + 1, start + contextLength + 1));
  }
  // Turn the inputs and targets into tensors of shape [batch, length] with dtype int32
  return {
    inputs: tf.tensor2d(inputs, [batchSize, contextLength], 'int32'),
    targets: tf.tensor2d(targets, [batchSize, contextLength], 'int32'),
  };
}

function generateCharacter() {
  // Get the last contextLength characters of what we've generated so far, and turn them into character ids
  let context = encode(generated.slice(-contextLength));
  // Ask the model for probabilities for the next character
  let probabilities = gpt.nextCharacter(context, temperature);
  // Pick a character at random, weighted by the probabilities
  let next = vocab[pickIndex(probabilities)];
  // Add it to the generated text
  generated += next;

  let lastLine = lines[lines.length - 1];
  if (next === '\n') {
    // A newline starts a new, empty line
    lines.push({ step: steps, text: '' });
  } else if (lastLine.text.length > 85) {
    // A line that's too long wraps, starting a new line with this character
    lines.push({ step: steps, text: next });
  } else {
    // Otherwise, add the character to the current line
    lastLine.text += next;
  }
}

// Weighted random selection (just like the Markov chain!)
function pickIndex(probabilities) {
  let r = random(1);
  for (let i = 0; i < probabilities.length; i++) {
    r -= probabilities[i];
    if (r <= 0) return i;
  }
  return probabilities.length - 1;
}

// Draw the loss over time
function drawLosses(x, y, w, h) {
  // Guessing at random among every character gives a loss of log(vocabulary size)
  let maxLoss = log(vocab.length);
  noStroke();
  fill(0);
  textSize(14);
  textAlign(LEFT, BOTTOM);
  let status = training ? 'training' : 'paused';
  let latest = nf(losses[losses.length - 1], 1, 2);
  text(`step ${steps}   loss ${latest}   ${status}`, x, y - 10);

  stroke(200);
  noFill();
  rect(x, y, w, h);
  stroke(0, 120, 255);
  beginShape();
  for (let i = 0; i < losses.length; i++) {
    // Spread the steps across the width of the graph
    let px = map(i, 0, max(losses.length - 1, 1), x, x + w);
    // A loss of 0 is at the bottom, maxLoss is at the top
    let py = map(losses[i], 0, maxLoss, y + h, y);
    vertex(px, py);
  }
  endShape();
}

// Draw the most recent lines the model has generated
function drawWriting(x, y) {
  noStroke();
  textAlign(LEFT, TOP);
  textSize(13);
  // Last 18 lines
  let recent = lines.slice(-18);
  for (let i = 0; i < recent.length; i++) {
    fill(150);
    text(`step ${recent[i].step}`, x, y + i * 15);
    fill(0);
    text(recent[i].text, x + 85, y + i * 15);
  }
}

// Draw the model's context window: the last characters it can see.
// Green shows how much attention one character pays to the characters before it.
function drawAttention(x, y) {
  let context = encode(generated.slice(-contextLength));
  let weights = gpt.attentionWeights(context);
  let boxSize = 23;

  // Which character is the mouse over
  let from = context.length - 1;
  for (let i = 0; i < context.length; i++) {
    if (mouseX > x + i * boxSize && mouseX < x + (i + 1) * boxSize && mouseY > y && mouseY < y + boxSize) {
      from = i;
    }
  }

  noStroke();
  fill(0);
  textSize(14);
  textAlign(LEFT, BOTTOM);
  text('attention:', x, y - 10);

  textAlign(CENTER, CENTER);
  for (let i = 0; i < context.length; i++) {
    // Get the attention weight from the "from" character to this character
    let weight = weights[from][i];
    // Set alpha according to attention weight
    fill(0, 255, 0, sqrt(weight) * 255);
    stroke(0);
    rect(x + i * boxSize, y, boxSize - 2, boxSize);
    noStroke();
    fill(0);
    let char = vocab[context[i]];
    text(char, x + i * boxSize + boxSize / 2, y + boxSize / 2);
  }
}
