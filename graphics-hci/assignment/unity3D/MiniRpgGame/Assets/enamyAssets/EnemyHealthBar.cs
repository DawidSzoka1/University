using playerAssets.FinalCharacterController;
using UnityEngine;
using UnityEngine.UI;

public class EnemyHealthBar : MonoBehaviour
{
 
    public Slider healthSlider;
    public Slider easeHealthSlider;
    public EnemyAiTutorial enemyAi;
    private float learpSpeed = 0.05f;

    void Update()
    {
        if (enemyAi.health != healthSlider.value)
        {
            healthSlider.value = enemyAi.health;
        }

        if (healthSlider.value != easeHealthSlider.value)
        {
            easeHealthSlider.value = Mathf.Lerp(easeHealthSlider.value, enemyAi.health, learpSpeed);
        }
    }
}
