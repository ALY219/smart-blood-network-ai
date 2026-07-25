import React, { useState, useRef } from 'react';
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

  // Ref for automatic smooth scrolling to newest messages
  const scrollViewRef = useRef();

  const handleSendMessage = async () => {
    if (!inputText.trim() || isLoading) return;

    const userMessage = inputText.trim();
    setInputText('');
    
    // 1. Add user message to chat UI
    setMessages(prev => [...prev, { id: Date.now().toString(), text: userMessage, isUser: true }]);
    setIsLoading(true);

    try {
      // 2. Call FastAPI backend service
      const backendResponse = await askBloodAgent(userMessage);
      
      // 3. Extract agent reply safely
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
      {/* Header */}
      <View style={styles.header}>
        <Text style={styles.headerTitle}>🩸 Smart Blood Agent</Text>
      </View>

      <KeyboardAvoidingView 
        style={styles.keyboardContainer}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
      >
        {/* Chat Feed with Auto-Scroll */}
        <ScrollView 
          ref={scrollViewRef}
          style={styles.chatArea} 
          contentContainerStyle={styles.chatContent}
          onContentSizeChange={() => scrollViewRef.current?.scrollToEnd({ animated: true })}
        >
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

        {/* Fixed Input Bar */}
        <View style={styles.inputContainer}>
          <TextInput
            style={styles.input}
            placeholder="Ask about donors, blood types..."
            placeholderTextColor="#95a5a6"
            value={inputText}
            onChangeText={setInputText}
            editable={!isLoading}
            onSubmitEditing={handleSendMessage}
            returnKeyType="send"
          />
          <TouchableOpacity 
            style={[styles.sendButton, (isLoading || !inputText.trim()) && styles.disabledButton]} 
            onPress={handleSendMessage}
            disabled={isLoading || !inputText.trim()}
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
    backgroundColor: '#f8f9fa',
  },
  header: {
    paddingVertical: 16,
    paddingHorizontal: 20,
    backgroundColor: '#e74c3c',
    alignItems: 'center',
    justifyContent: 'center',
    elevation: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.2,
    shadowRadius: 3,
  },
  headerTitle: {
    color: '#fff',
    fontSize: 19,
    fontWeight: 'bold',
  },
  keyboardContainer: {
    flex: 1,
  },
  chatArea: {
    flex: 1,
  },
  chatContent: {
    padding: 16,
    paddingBottom: 20,
  },
  bubble: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    borderRadius: 16,
    marginBottom: 12,
    maxWidth: '82%',
  },
  userBubble: {
    backgroundColor: '#e74c3c',
    alignSelf: 'flex-end',
    borderBottomRightRadius: 2,
  },
  agentBubble: {
    backgroundColor: '#eaeded',
    alignSelf: 'flex-start',
    borderBottomLeftRadius: 2,
  },
  userText: {
    color: '#ffffff',
    fontSize: 15,
    lineHeight: 22,
  },
  agentText: {
    color: '#2c3e50',
    fontSize: 15,
    lineHeight: 22,
  },
  loadingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    padding: 10,
    alignSelf: 'flex-start',
    backgroundColor: '#f2f4f4',
    borderRadius: 16,
    marginBottom: 12,
  },
  loadingText: {
    marginLeft: 8,
    color: '#7f8c8d',
    fontSize: 14,
    fontStyle: 'italic',
  },
  inputContainer: {
    flexDirection: 'row',
    padding: 12,
    backgroundColor: '#ffffff',
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
    color: '#2c3e50',
  },
  sendButton: {
    marginLeft: 8,
    backgroundColor: '#e74c3c',
    borderRadius: 22,
    paddingVertical: 11,
    paddingHorizontal: 20,
    justifyContent: 'center',
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#f5b7b1',
  },
  sendButtonText: {
    color: '#ffffff',
    fontWeight: 'bold',
    fontSize: 15,
  },
});