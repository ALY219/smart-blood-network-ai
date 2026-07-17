import React, { useState, useEffect } from 'react';
import { StyleSheet, Text, View, TextInput, TouchableOpacity, Alert, ActivityIndicator, KeyboardAvoidingView, Platform, FlatList } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import MapView, { Marker } from 'react-native-maps';
import { collection, onSnapshot, query, orderBy } from 'firebase/firestore';
import { db } from './firebase'; 

// FIXED: Using your exact local IP address
const API_URL = 'http://192.168.18.59:8001';
const Stack = createNativeStackNavigator();

// --- 1. EMERGENCY DISPATCHER COMPONENT ---
function EmergencyScreen({ navigation }) {
  const [patientName, setPatientName] = useState('Ahmad');
  const [bloodGroup, setBloodGroup] = useState('A+');
  const [isLoading, setIsLoading] = useState(false);

  const triggerEmergency = async () => {
    setIsLoading(true);
    try {
      // FIXED: Corrected BACKEND_URL to API_URL
      const response = await fetch(`${API_URL}/api/v1/emergency-request`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_name: patientName,
          blood_group: bloodGroup,
          units_required: 2,
          hospital_name: "Jinnah Hospital Lahore",
          emergency_notes: "Critical condition",
          latitude: 31.5204,
          longitude: 74.3587
        }),
      });
      
      if (response.ok) {
        navigation.replace('LiveMap');
      } else {
        Alert.alert("Error", "Backend responded with an error.");
      }
    } catch (error) {
      console.error(error);
      Alert.alert("Connection Error", "Ensure your server is running and reachable at " + API_URL);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView behavior={Platform.OS === 'ios' ? 'padding' : 'height'} style={styles.container}>
      <View style={styles.card}>
        <Text style={styles.title}>Smart Blood Network</Text>
        <Text style={styles.subtitle}>Autonomous AI Dispatcher</Text>
        <TextInput style={styles.input} placeholder="Patient Name" value={patientName} onChangeText={setPatientName} />
        <TextInput style={styles.input} placeholder="Blood Group" value={bloodGroup} onChangeText={setBloodGroup} autoCapitalize="characters" />
        <TouchableOpacity style={styles.emergencyButton} onPress={triggerEmergency} disabled={isLoading}>
          {isLoading ? <ActivityIndicator color="#fff" /> : <Text style={styles.buttonText}>🚨 TRIGGER EMERGENCY</Text>}
        </TouchableOpacity>
      </View>
    </KeyboardAvoidingView>
  );
}

// --- 2. LIVE DONOR TRACKER & AI LOG FEED ---
function LiveMapScreen() {
  const [donors, setDonors] = useState([]);
  const [logs, setLogs] = useState([]);

  useEffect(() => {
    const dSub = onSnapshot(collection(db, 'donors'), (s) => setDonors(s.docs.map(d => ({ id: d.id, ...d.data() }))));
    const lSub = onSnapshot(query(collection(db, 'logs'), orderBy("timestamp", "desc")), (s) => setLogs(s.docs.map(d => d.data())));
    return () => { dSub(); lSub(); };
  }, []);

  return (
    <View style={styles.mapContainer}>
      <MapView style={styles.map} initialRegion={{ latitude: 31.5204, longitude: 74.3587, latitudeDelta: 0.02, longitudeDelta: 0.02 }}>
        {donors.map(d => <Marker key={d.id} coordinate={{ latitude: d.lat, longitude: d.lng }} title={d.name} pinColor="blue" />)}
      </MapView>
      
      <View style={styles.overlayCard}>
        <Text style={styles.overlayTitle}>AI Reasoning Feed</Text>
        <FlatList 
          data={logs}
          keyExtractor={(_, i) => i.toString()}
          renderItem={({item}) => <Text style={styles.logText}>✅ {item.message}</Text>}
          style={{ maxHeight: 150 }}
        />
      </View>
    </View>
  );
}

export default function App() {
  return (
    <NavigationContainer>
      <Stack.Navigator screenOptions={{ headerShown: false }}>
        <Stack.Screen name="Dispatch" component={EmergencyScreen} />
        <Stack.Screen name="LiveMap" component={LiveMapScreen} />
      </Stack.Navigator>
    </NavigationContainer>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#f5f5f5', justifyContent: 'center', padding: 20 },
  card: { backgroundColor: 'white', padding: 24, borderRadius: 16, elevation: 5 },
  title: { fontSize: 22, fontWeight: 'bold', color: '#1f2937', textAlign: 'center' },
  subtitle: { fontSize: 14, color: '#6b7280', textAlign: 'center', marginBottom: 20 },
  input: { backgroundColor: '#f9fafb', borderWidth: 1, borderColor: '#e5e7eb', borderRadius: 8, padding: 16, marginBottom: 16 },
  emergencyButton: { backgroundColor: '#ef4444', padding: 18, borderRadius: 8, alignItems: 'center' },
  buttonText: { color: 'white', fontWeight: 'bold' },
  mapContainer: { flex: 1 },
  map: { width: '100%', height: '100%' },
  overlayCard: { position: 'absolute', bottom: 30, left: 20, right: 20, backgroundColor: 'white', padding: 15, borderRadius: 12, elevation: 5 },
  overlayTitle: { fontSize: 16, fontWeight: 'bold', color: '#374151', marginBottom: 10 },
  logText: { fontSize: 12, color: '#4b5563', marginBottom: 4 }
});