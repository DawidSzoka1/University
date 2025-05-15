using UnityEngine;


namespace playerAssets.FinalCharacterController
{
    public class PlayerState : MonoBehaviour
    {
        [field: SerializeField] public PlayerMovementState CurrrentPlayerMovementState { get; private set; } = PlayerMovementState.Idling;
        
        public void SetPlayerMovementState(PlayerMovementState playerMovementState)
        {
            CurrrentPlayerMovementState = playerMovementState;
        }

        public bool InGroundedState()
        {
            return CurrrentPlayerMovementState != PlayerMovementState.Jumping &&
                CurrrentPlayerMovementState != PlayerMovementState.Falling;
        }
        
       
    }
    public enum PlayerMovementState
    {
        Idling = 0,
        Walking = 1,
        Running = 2,
        Sprinting = 3,
        Jumping = 4,
        Falling = 5,
        Strafing = 6,
    }

}
