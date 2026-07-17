import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";

// Your exact Firebase configuration
const firebaseConfig = {
  apiKey: "AIzaSyAJaBDlCNuskIeWhzKylFu800D37ZICr0Q",
  authDomain: "smart-blood-network-6a2f6.firebaseapp.com",
  projectId: "smart-blood-network-6a2f6",
  storageBucket: "smart-blood-network-6a2f6.firebasestorage.app",
  messagingSenderId: "130789238727",
  appId: "1:130789238727:web:eaaffd0399224bb2cc1be2",
  measurementId: "G-S7BKJ6VW98"
};

// Initialize Firebase and export the Database connection
const app = initializeApp(firebaseConfig);
export const db = getFirestore(app);