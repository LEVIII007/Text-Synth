function handleImageUpload(event) {
    try {
        const file = event.target.files[0];
        const reader = new FileReader();
        
        reader.onload = function() {
            try {
                const arrayBuffer = this.result;
                const image = document.createElement('img');
                image.src = URL.createObjectURL(new Blob([arrayBuffer]));
                image.classList.add('uploaded-image');
                
                const outputDiv = document.getElementById('outputContainer');
                const newOutput = document.createElement('div');
                newOutput.classList.add('output-item');
                
                const imageBlock = document.createElement('div');
                imageBlock.classList.add('image-block');
                imageBlock.appendChild(image);
                
                newOutput.appendChild(imageBlock);
                
                const timestamp = document.createElement('div');
                timestamp.textContent = `Uploaded at: ${new Date().toLocaleString()}`;
                newOutput.appendChild(timestamp);
                
                outputDiv.appendChild(newOutput);

                const socket = new WebSocket(`ws://${window.location.host}/ws/socket-server/`);
                console.log('WebSocket connection created: --------------------------------------------------', socket);
                console.log('Image:', image.src);
                // Send the image data over the WebSocket connection
                socket.send(arrayBuffer);

                // Rest of your code...
            } catch (error) {
                console.error('Error in reader.onload:', error);
            }
        };
        
        reader.readAsArrayBuffer(file);
    } catch (error) {
        console.error('Error in handleImageUpload:', error);
    }
}


function handleSubmit(event) {
    try {
        event.preventDefault();
        
        const images = document.querySelectorAll('.uploaded-image');
    } catch (error) {
        console.error('Error in handleSubmit:', error);
    }
}