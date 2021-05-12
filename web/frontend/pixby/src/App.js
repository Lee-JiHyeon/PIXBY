import "./App.css";
import styled from "styled-components";
import pixby from "./img/pixby.png";

const Title = styled.div`
  font-size: 280px;
  margin: auto;
`;

function App() {
  return (
    <div className="main">
      <Title>
        <img src={pixby} />
      </Title>
      <button>
        {/* <a href={pixby} download> */}
        <a href={"./ex.pdf"} type="application/pdf" download="some_pdf_name">
          Click to download
        </a>
      </button>
    </div>
  );
}

export default App;
