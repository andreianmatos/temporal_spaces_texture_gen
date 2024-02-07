class CVAE {
    constructor(latent_dim, encoder, decoder) {
        this.latent_dim = latent_dim;
        this.encoder = encoder;
        this.decoder = decoder;
    }

    async sample(eps) {
        if (eps === undefined) {
            eps = tf.randomNormal([100, this.latent_dim]);
        }
        return this.decode(eps, true);
    }

    async encode(x) {
        const [mean, logvar] = tf.split(this.encoder.predict(x), 2, 1);
        return [mean, logvar];
    }

    reparameterize(mean, logvar) {
        const eps = tf.randomNormal(mean.shape);
        return tf.add(tf.mul(tf.exp(tf.mul(logvar, 0.5)), eps), mean);
    }

    async decode(z, apply_sigmoid=false) {
        const logits = this.decoder.predict(z);
        if (apply_sigmoid) {
            return tf.sigmoid(logits);
        }
        return logits;
    }
}

async function loadModels() {
    // Load the encoder and decoder models

    const decoder = await tf.loadGraphModel('model_json/captures/500_epochs/decoder/model.json');
    const encoder = await tf.loadGraphModel('model_json/captures/500_epochs/encoder/model.json');
    
    const latent_dim = 2;
    const loaded_model = new CVAE(latent_dim, encoder, decoder);

    // Generate and display images
    const numExamplesToGenerate = 10; // Change the desired number of examples
    const latentDim = 2;
    const generatedImages = [];
    
    for (let i = 0; i < numExamplesToGenerate; i++) {
      const randomVectorForGeneration = tf.randomNormal([1, latentDim]); // Generate a new random vector for each example
      const sample = await loaded_model.sample(randomVectorForGeneration);
      const generatedImage = await generateImage(loaded_model, sample);
      console.log(generatedImage);
      generatedImages.push(generatedImage);
    }


    // Display the generated images
    const imageContainer = document.getElementById('imageContainer');
    generatedImages.forEach((imageData, index) => {
        const canvas = document.createElement('canvas');
        canvas.width = imageData.width;
        canvas.height = imageData.height;
        const context = canvas.getContext('2d');
        context.putImageData(imageData, 0, 0);

        const img = document.createElement('img');
        img.src = canvas.toDataURL();
        img.alt = `Generated Image ${index + 1}`;
        img.classList.add('generatedImage'); 
        imageContainer.appendChild(img);
    });
}

async function generateImage(model, sample) {
    const [mean, logvar] = await model.encode(sample);
    const z = model.reparameterize(mean, logvar);
    const prediction = await model.sample(z);
    // Convert the tensor to pixels
    const pixelsTensor = prediction.squeeze();
    const pixels = await tf.browser.toPixels(pixelsTensor);

    // Create an ImageData object from the pixel data
    const image = new ImageData(new Uint8ClampedArray(pixels), pixelsTensor.shape[1], pixelsTensor.shape[0]);

    return image;
}

// Utility function to convert ArrayBuffer to Base64
function arrayBufferToBase64(buffer) {
    let binary = '';
    const bytes = new Uint8Array(buffer);
    const len = bytes.byteLength;
    for (let i = 0; i < len; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return window.btoa(binary);
}

loadModels();
