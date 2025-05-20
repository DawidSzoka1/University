using UnityEngine;
using System.Collections.Generic;
using System.Collections;

public class EnemySpawner : MonoBehaviour
{

    public GameObject enemyPrefab;
    public int maxEnemies = 5;
    public float spawnWidth = 50f;  
    public float spawnDepth = 50f;

    private List<GameObject> activeEnemies = new List<GameObject>();

    void Start()
    {
        for (int i = 0; i < maxEnemies; i++)
        {
            SpawnEnemy();
        }
    }

    public void SpawnEnemy()
    {
        Vector3 center = transform.position;

        float spawnX = Random.Range(center.x - spawnWidth / 2, center.x + spawnWidth / 2);
        float spawnZ = Random.Range(center.z - spawnDepth / 2, center.z + spawnDepth / 2);
        float spawnY = center.y;

        Vector3 spawnPos = new Vector3(spawnX, spawnY, spawnZ);

        GameObject enemy = Instantiate(enemyPrefab, spawnPos, Quaternion.identity);
        enemy.GetComponent<EnemyAiTutorial>().SetSpawner(this);
        activeEnemies.Add(enemy);
    }

    public void OnEnemyDeath(GameObject enemy)
    {
        activeEnemies.Remove(enemy);
        StartCoroutine(RespawnAfterDelay(30f));
    }

    private IEnumerator RespawnAfterDelay(float delay)
    {
        yield return new WaitForSeconds(delay);
        SpawnEnemy();
    }
    public void OnDrawGizmosSelected()
    {
        Gizmos.color = Color.red;
        Gizmos.DrawWireCube(transform.position, new Vector3(spawnWidth, 0.1f, spawnDepth));
    }
}
