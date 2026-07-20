import React, { useState } from 'react';
import { 
  StyleSheet, 
  Text, 
  View, 
  TextInput, 
  TouchableOpacity, 
  ScrollView, 
  ActivityIndicator, 
  SafeAreaView, 
  KeyboardAvoidingView, 
  Platform 
} from 'react-native';
import { askBloodAgent } from './services/api';

export default function App() {
  const [messages, setMessages] = useState([
    { id: '1', text: 'Hello! I am your Smart Blood Network Assistant. How can I help you today?', isUser: false }
  ]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = inputText.trim();
    setInputText('');
    
    // 1. Add user message to chat UI
    setMessages(prev => [...prev, { id: Date.now().toString(), text: userMessage, isUser: true }]);
    setIsLoading(true);

    try {
      // 2. Call our FastAPI backend service
      const backendResponse = await askBloodAgent(userMessage);
      
      // 3. Extract the agent's reply (adjust key based on your FastAPI response schema)
      const agentReply = backendResponse.reply || backendResponse.message || JSON.stringify(backendResponse);

      // 4. Add AI response to chat UI
      setMessages(prev => [...prev, { id: (Date.now() + 1).toString(), text: agentReply, isUser: false }]);
    } catch (error) {
      setMessages(prev => [...prev, { 
        id: (Date.now() + 1).toString(), 
        text: '⚠️ Failed to connect to the AI agent. Please check your backend connection.', 
        isUser: false 
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <SafeAreaView style={styles.container}>
      <View style={styles.header}>
        <Text style={styles.headerTitle}>🩸 Smart Blood Agent</Text>
      </View>

      <ScrollView style={styles.chatArea} contentContainerStyle={{ paddingBottom: 20 }}>
        {messages.map((msg) => (
          <View 
            key={msg.id} 
            style={[styles.bubble, msg.isUser ? styles.userBubble : styles.agentBubble]}
          >
            <Text style={msg.isUser ? styles.userText : styles.agentText}>
              {msg.text}
            </Text>
          </View>
        ))}
        
        {isLoading && (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="small" color="#e74c3c" />
            <Text style={styles.loadingText}>Agent is analyzing...</Text>
          </View>
        )}
      </ScrollView>

      <KeyboardAvoidingView 
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 90 : 0}
      >
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="Ask about donors, blood types..."
            value={inputText}
            onChangeText={setInputText}
            editable={!isLoading}
          />
          <TouchableOpacity 
            style={[styles.sendButton, isLoading && styles.disabledButton]} 
            onPress={handleSendMessage}
            disabled={isLoading}
          >
            <Text style={styles.sendButtonText}>Send</Text>
          </TouchableOpacity>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#f9f9f9',
  },
  header: {
    padding: 16,
    backgroundColor: '#e74c3c',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 18,
    fontWeight: 'bold',
  },
  chatArea: {
    flex: 1,
    padding: 16,
  },
  bubble: {
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
    maxWidth: '80%',
  },
  userBubble: {
    backgroundColor: '#e74c3c',
    alignSelf: 'flex-end',
    borderBottomRightRadius: 0,
  },
  agentBubble: {
    backgroundColor: '#eaeded',
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 0,
  },
  userText: {
    color: '#fff',
    fontSize: 15,
  },
  agentText: {
    color: '#2c3e50',
    fontSize: 15,
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
  },
  loadingText: {
    marginLeft: 8,
    color: '#7f8c8d',
    fontStyle: 'italic',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 12,
    backgroundColor: '#fff',
    borderTopWidth: 1,
    borderTopColor: '#e5e7e9',
    alignItems: 'center',
  },
  input: {
    flex: 1,
    height: 44,
    borderWidth: 1,
    borderColor: '#d5dbdb',
    borderRadius: 22,
    paddingHorizontal: 16,
    backgroundColor: '#f8f9f9',
    fontSize: 15,
  },
  sendButton: {
    marginLeft: 8,
    backgroundColor: '#e74c3c',
    borderRadius: 22,
    paddingVertical: 10,
    paddingHorizontal: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#f5b7b1',
  },
  sendButtonText: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: 15,
  },
});