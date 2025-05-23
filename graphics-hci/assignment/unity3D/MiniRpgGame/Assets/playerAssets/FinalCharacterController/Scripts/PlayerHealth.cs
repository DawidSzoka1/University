using System.Runtime.CompilerServices;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.SceneManagement;
using UnityEngine.UI;
namespace playerAssets.FinalCharacterController
{

    public class PlayerHealth : MonoBehaviour
    {
        public int maxHealth = 100;
        public int currentHealth;
        public int healthPerKill = 20;
        public GameObject gameOverUI;
        [SerializeField] private Animator _animator;
      

        void Start()
        {
            currentHealth = maxHealth;
            gameOverUI.SetActive(false);
        }
        
        public void TakeDamage(int amount)
        {
            currentHealth -= amount;
            currentHealth = Mathf.Max(currentHealth, 0);

       
            Debug.Log("Gracz otrzyma� " + amount + " obra�e�. HP = " + currentHealth);
            Invoke(nameof(AnimationOnHit), 0.5f);
            
            if (currentHealth <= 0)
            {
                EndGame();
            }
        }

        public void HealAfterKill()
        {
            currentHealth += healthPerKill;
            if (currentHealth >= maxHealth)
            {
                currentHealth = maxHealth;
            }
        }

        private void AnimationOnHit()
        {
            _animator.SetTrigger("getHit");
        }

        private void EndGame()
        {
            Time.timeScale = 0f; 
            gameOverUI.SetActive(true); 
            Cursor.lockState = CursorLockMode.None;
            Cursor.visible = true;
        }

        public void RestartGame()
        {
            Time.timeScale = 1f;
            SceneManager.LoadScene(SceneManager.GetActiveScene().buildIndex);
        }

        public void ExitGame()
        {
            Application.Quit();
        }

    }
}

