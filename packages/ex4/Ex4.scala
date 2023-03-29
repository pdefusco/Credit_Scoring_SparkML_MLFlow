package bobsrockets {
  package navigation {
    class Navigator
  }
  package launch {
    class Booster {
      // No need to say bobsrockets.navigation.Navigator
      val nav = new navigation.Navigator
    }
  }
}