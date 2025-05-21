using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.UI;

namespace playerAssets.FinalCharacterController
{
    public class HealthBar : MonoBehaviour
    {
        public Slider healthSlider;
        public Slider easeHealthSlider;
        public PlayerHealth playerHealth;
        private float learpSpeed = 0.05f;

        void Update()
        {
            if(playerHealth.currentHealth !=  healthSlider.value) 
            {
                healthSlider.value = playerHealth.currentHealth;
            }

            if(healthSlider.value != easeHealthSlider.value)
            {
                easeHealthSlider.value = Mathf.Lerp(easeHealthSlider.value, playerHealth.currentHealth, learpSpeed);
            }
        }
    }

}
