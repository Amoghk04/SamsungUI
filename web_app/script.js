document.addEventListener("DOMContentLoaded", () => {
    const sendButton = document.getElementById("send-button");
    const userInput = document.getElementById("user-input");
    const chatMessages = document.getElementById("chat-messages");

    sendButton.addEventListener("click", async () => {
        const userMessage = userInput.value.trim();
        if (!userMessage) return;

        // Display user's message.
        addMessage(userMessage, "user");

        try {
            // Send the message to the backend.
            const response = await fetch("http://127.0.0.1:8000/predict/", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ sentence: userMessage }),
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const responseData = await response.json();

            // Build the assistant's message.
            const assistantMessage = `Intent: ${responseData.intent}\nSlots:\n` +
                Object.entries(responseData.slots)
                    .map(([token, slot]) => `${token}: ${slot}`)
                    .join("\n");
            addMessage(assistantMessage, "assistant");

        } catch (error) {
            console.error("Error during fetch:", error);
            addMessage("Error: Could not reach the backend.", "assistant");
        }

        // Clear input field.
        userInput.value = "";
    });

    function addMessage(text, sender) {
        const messageDiv = document.createElement("div");
        messageDiv.classList.add("message", sender);

        const bubbleDiv = document.createElement("div");
        bubbleDiv.classList.add("bubble");
        bubbleDiv.textContent = text;

        messageDiv.appendChild(bubbleDiv);
        chatMessages.appendChild(messageDiv);

        // Scroll to the bottom.
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});